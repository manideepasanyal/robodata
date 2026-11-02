// High-rate stamp logger + live sync monitor.
//
// Subscribes to ANY stamped topic through rclcpp::GenericSubscription and reads the
// std_msgs/Header stamp straight out of the serialized CDR buffer (no deserialization),
// so it keeps up with kHz force-torque / joint streams and full-resolution images alike.
// For every message it logs (t_device, t_host, size) to a compact binary file per topic and
// publishes per-topic rate, header->receive latency percentiles, max gap and clock steps on
// /diagnostics. robodata.io.stamps.read_stamp_log() loads the files.
//
// CDR layout for messages whose first field is std_msgs/Header:
//   [0..3] encapsulation header (0x00 0x01 = CDR little endian), [4..7] int32 sec, [8..11] uint32 nanosec

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialized_message.hpp"

namespace
{
#pragma pack(push, 1)
struct StampRecord
{
  int64_t t_device_ns;
  int64_t t_host_ns;
  uint32_t bytes;
};
#pragma pack(pop)

bool read_header_stamp(const rcl_serialized_message_t & buf, int64_t & out_ns)
{
  if (buf.buffer_length < 12) {
    return false;
  }
  const uint8_t * b = buf.buffer;
  const bool little = b[1] == 0x01;  // CDR_LE (0x0001) vs CDR_BE (0x0000)
  int32_t sec;
  uint32_t nsec;
  std::memcpy(&sec, b + 4, 4);
  std::memcpy(&nsec, b + 8, 4);
  if (!little) {
    sec = static_cast<int32_t>(__builtin_bswap32(static_cast<uint32_t>(sec)));
    nsec = __builtin_bswap32(nsec);
  }
  out_ns = static_cast<int64_t>(sec) * 1000000000LL + nsec;
  return true;
}

double percentile(std::vector<double> v, double q)
{
  if (v.empty()) {
    return 0.0;
  }
  const size_t k = static_cast<size_t>(q * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}
}  // namespace

class StampMonitor : public rclcpp::Node
{
public:
  StampMonitor()
  : Node("robodata_stamp_monitor")
  {
    const auto topics = declare_parameter<std::vector<std::string>>("topics", std::vector<std::string>{});
    const auto types = declare_parameter<std::vector<std::string>>("types", std::vector<std::string>{});
    const auto out_dir = declare_parameter<std::string>("output_dir", "/tmp/robodata_stamps");
    window_ = static_cast<size_t>(declare_parameter<int>("window", 2000));
    jump_ms_ = declare_parameter<double>("clock_step_threshold_ms", 20.0);
    if (topics.size() != types.size()) {
      throw std::invalid_argument("'topics' and 'types' must have the same length");
    }
    std::filesystem::create_directories(out_dir);
    for (size_t i = 0; i < topics.size(); ++i) {
      auto s = std::make_shared<Stream>();
      s->topic = topics[i];
      std::string fname = topics[i];
      std::replace(fname.begin(), fname.end(), '/', '_');
      s->file.open(out_dir + "/" + fname + ".rdts", std::ios::binary | std::ios::trunc);
      s->file.write("RDTS1\n", 6);
      streams_.push_back(s);
      auto qos = rclcpp::SensorDataQoS().keep_last(500);
      subs_.push_back(create_generic_subscription(
          topics[i], types[i], qos,
          [this, s](std::shared_ptr<const rclcpp::SerializedMessage> msg) {on_message(*s, *msg);}));
      RCLCPP_INFO(get_logger(), "monitoring %s [%s]", topics[i].c_str(), types[i].c_str());
    }
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {publish_diagnostics();});
  }

  ~StampMonitor() override
  {
    for (auto & s : streams_) {
      std::lock_guard<std::mutex> lk(s->mtx);
      s->file.flush();
    }
  }

private:
  struct Stream
  {
    std::string topic;
    std::ofstream file;
    std::mutex mtx;
    std::deque<StampRecord> recent;
    uint64_t count = 0;
    uint64_t clock_steps = 0;
    uint64_t non_monotonic = 0;
    int64_t last_offset_ns = 0;
    int64_t last_device_ns = 0;
  };

  void on_message(Stream & s, const rclcpp::SerializedMessage & msg)
  {
    const int64_t t_host = now().nanoseconds();
    int64_t t_dev = 0;
    if (!read_header_stamp(msg.get_rcl_serialized_message(), t_dev)) {
      return;
    }
    StampRecord r{t_dev, t_host, static_cast<uint32_t>(msg.size())};
    std::lock_guard<std::mutex> lk(s.mtx);
    s.file.write(reinterpret_cast<const char *>(&r), sizeof(r));
    if (s.count > 0) {
      if (t_dev <= s.last_device_ns) {
        ++s.non_monotonic;
      }
      // a persistent change of (host - device) larger than jitter => device clock stepped
      const int64_t offset = t_host - t_dev;
      if (std::abs(static_cast<double>(offset - s.last_offset_ns)) > jump_ms_ * 1e6 &&
        std::abs(static_cast<double>(offset - min_offset(s))) > jump_ms_ * 1e6)
      {
        ++s.clock_steps;
      }
      s.last_offset_ns = offset;
    } else {
      s.last_offset_ns = t_host - t_dev;
    }
    s.last_device_ns = t_dev;
    ++s.count;
    s.recent.push_back(r);
    if (s.recent.size() > window_) {
      s.recent.pop_front();
    }
  }

  static int64_t min_offset(const Stream & s)
  {
    int64_t m = INT64_MAX;
    for (const auto & r : s.recent) {
      m = std::min(m, r.t_host_ns - r.t_device_ns);
    }
    return m;
  }

  void publish_diagnostics()
  {
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    for (auto & s : streams_) {
      std::lock_guard<std::mutex> lk(s->mtx);
      s->file.flush();
      diagnostic_msgs::msg::DiagnosticStatus st;
      st.name = "robodata_stamps" + s->topic;
      st.hardware_id = s->topic;
      if (s->recent.size() < 2) {
        st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        st.message = "no data";
        arr.status.push_back(st);
        continue;
      }
      std::vector<double> lat, gaps;
      for (size_t i = 0; i < s->recent.size(); ++i) {
        lat.push_back((s->recent[i].t_host_ns - s->recent[i].t_device_ns) / 1e6);
        if (i) {
          gaps.push_back((s->recent[i].t_device_ns - s->recent[i - 1].t_device_ns) / 1e6);
        }
      }
      const double span = (s->recent.back().t_device_ns - s->recent.front().t_device_ns) / 1e9;
      const double rate = span > 0 ? (s->recent.size() - 1) / span : 0.0;
      auto kv = [&](const std::string & k, double v) {
          diagnostic_msgs::msg::KeyValue x;
          x.key = k;
          x.value = std::to_string(v);
          st.values.push_back(x);
        };
      kv("rate_hz", rate);
      kv("latency_p50_ms", percentile(lat, 0.5));
      kv("latency_p95_ms", percentile(lat, 0.95));
      kv("max_gap_ms", *std::max_element(gaps.begin(), gaps.end()));
      kv("clock_steps", static_cast<double>(s->clock_steps));
      kv("non_monotonic", static_cast<double>(s->non_monotonic));
      kv("received", static_cast<double>(s->count));
      st.level = (s->clock_steps || s->non_monotonic) ? diagnostic_msgs::msg::DiagnosticStatus::WARN :
        diagnostic_msgs::msg::DiagnosticStatus::OK;
      st.message = std::to_string(rate) + " Hz";
      arr.status.push_back(st);
    }
    diag_pub_->publish(arr);
  }

  std::vector<std::shared_ptr<Stream>> streams_;
  std::vector<rclcpp::GenericSubscription::SharedPtr> subs_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  size_t window_{2000};
  double jump_ms_{20.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StampMonitor>();
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
