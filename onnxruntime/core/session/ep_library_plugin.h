// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <mutex>

#include "core/session/ep_library.h"
#include "core/session/ep_library_provider_bridge.h"

namespace onnxruntime {
class EpLibraryPlugin : public EpLibrary {
 public:
  EpLibraryPlugin(const std::string& registration_name, const ORTCHAR_T* library_path)
      : registration_name_{registration_name},
        library_path_{library_path} {
  }

  const char* RegistrationName() const override {
    return registration_name_.c_str();
  }

  Status Load() override;

  const std::vector<OrtEpFactory*>& GetFactories() override {
    return factories_;
  }

  Status Unload() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryPlugin);

 private:
  std::mutex mutex_;
  const std::string registration_name_;
  const std::filesystem::path library_path_;
  void* handle_{};
  std::vector<OrtEpFactory*> factories_{};
  CreateEpApiFactoriesFn create_fn_{nullptr};
  ReleaseEpApiFactoryFn release_fn_{nullptr};
};
}  // namespace onnxruntime
