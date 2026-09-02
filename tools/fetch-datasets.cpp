#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/error.hpp"

int main(int argc, char** argv) {
  CLI::App app{"Download TurboVec/TurboQuant benchmark datasets into data/"};
  std::filesystem::path data_dir = "data";
  bool force = false;
  std::vector<std::string> targets;

  app.add_option("--data-dir", data_dir, "Output directory for downloaded files");
  app.add_flag("--force", force, "Re-download even when a valid file already exists");
  app.add_option("targets", targets, "Datasets: glove, openai-1536, openai-3072, or all")
      ->required();

  CLI11_PARSE(app, argc, argv);

  try {
    std::vector<vectorcache::datasets::DatasetKind> kinds;
    if (std::find(targets.begin(), targets.end(), "all") != targets.end()) {
      if (targets.size() > 1) {
        throw vectorcache::Error("pass either 'all' or explicit dataset names, not both");
      }
      kinds = vectorcache::datasets::all_dataset_kinds();
    } else {
      for (const auto& target : targets) {
        auto kind = vectorcache::datasets::parse_dataset_kind(target);
        if (!kind) {
          throw vectorcache::Error("unknown dataset '" + target + "'");
        }
        kinds.push_back(*kind);
      }
    }

    for (const auto kind : kinds) {
      std::cout << "Fetching " << vectorcache::datasets::dataset_label(kind) << " ...\n";
      vectorcache::datasets::fetch(kind, data_dir, force);
    }
    return 0;
  } catch (const vectorcache::Error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
