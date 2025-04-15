// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
// helper to forward a call from the C API to an instance of the factory implementation.
// used by EpFactoryInternal and EpFactoryProviderBridge.
template <typename TFactory>
struct ForwardToFactory {
  static const char* GetFactoryName(const OrtEpApi::OrtEpFactory* this_ptr) {
    return static_cast<const TFactory*>(this_ptr)->GetName();
  }

  static const char* GetVendor(const OrtEpApi::OrtEpFactory* this_ptr) {
    return static_cast<const TFactory*>(this_ptr)->GetVendor();
  }

  static bool GetDeviceInfoIfSupported(const OrtEpApi::OrtEpFactory* this_ptr,
                                       const OrtHardwareDevice* device,
                                       OrtKeyValuePairs** ep_device_metadata,
                                       OrtKeyValuePairs** ep_options_for_device) {
    return static_cast<const TFactory*>(this_ptr)->GetDeviceInfoIfSupported(device, ep_device_metadata,
                                                                            ep_options_for_device);
  }

  static OrtStatus* CreateEp(OrtEpApi::OrtEpFactory* this_ptr,
                             const OrtHardwareDevice* const* devices,
                             const OrtKeyValuePairs* const* ep_metadata_pairs,
                             size_t num_devices,
                             const OrtSessionOptions* session_options,
                             const OrtLogger* logger,
                             OrtEpApi::OrtEp** ep) {
    return static_cast<TFactory*>(this_ptr)->CreateEp(devices, ep_metadata_pairs, num_devices,
                                                      session_options, logger, ep);
  }

  static void ReleaseEp(OrtEpApi::OrtEpFactory* this_ptr, OrtEpApi::OrtEp* ep) {
    static_cast<TFactory*>(this_ptr)->ReleaseEp(ep);
  }
};
}  // namespace onnxruntime
