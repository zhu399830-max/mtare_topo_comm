#include <utils/planner_seed.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    return 2;
  }
  const auto seed = static_cast<std::uint32_t>(std::stoul(argv[1]));
  planner_seed_ns::Initialize(seed);
  const std::uint32_t streams[] = {
    planner_seed_ns::kLocalCoverageViewpointStream,
    planner_seed_ns::kGridWorldRobotSubsetStream,
    planner_seed_ns::kPBSTieBreakStream,
    planner_seed_ns::kCBSTieBreakStream,
  };
  std::uniform_int_distribution<std::uint32_t> distribution;
  std::cout << "seed=" << planner_seed_ns::BaseSeed() << "\n";
  for (const auto stream : streams)
  {
    std::cout << std::hex << stream << std::dec;
    auto& generator = planner_seed_ns::Generator(stream);
    for (int i = 0; i < 32; ++i)
    {
      std::cout << " " << distribution(generator);
    }
    std::cout << "\n";
  }
  return 0;
}
