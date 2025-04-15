#include "example_plugin_ep.h"
#include "onnxruntime_cxx_api.h"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include <gsl/span>

using OrtEpFactory = OrtEpApi::OrtEpFactory;
using OrtEp = OrtEpApi::OrtEp;

#define RETURN_IF_ERROR(fn)   \
  do {                        \
    OrtStatus* status = (fn); \
    if (status != nullptr) {  \
      return status;          \
    }                         \
  } while (0)

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

struct ExampleEp : OrtEp, ApiPtrs {
  ExampleEp(ApiPtrs apis, const std::string& name, const OrtSessionOptions& session_options, const OrtLogger& logger)
      : ApiPtrs(apis), name_{name}, session_options_{session_options}, logger_{logger} {
    // Initialize the execution provider.

    auto status = ort_api.Logger_LogMessage(&logger_,
                                            OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                            ("ExampleEp has been created with name " + name_).c_str(),
                                            ORT_FILE, __LINE__, __FUNCTION__);
    // ignore status for now
    (void*)status;
  }

  ~ExampleEp() {
    // Clean up the execution provider
  }

  std::string name_;
  const OrtSessionOptions& session_options_;
  const OrtLogger& logger_;
};

struct ExampleEpFactory : OrtEpFactory, ApiPtrs {
  ExampleEpFactory(const char* ep_name, ApiPtrs apis) : ApiPtrs(apis), ep_name_{ep_name} {
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetDeviceInfoIfSupported = GetDeviceInfoIfSupportedImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
  }

  static const char* GetNameImpl(const OrtEpFactory* this_ptr) {
    const auto* ep = static_cast<const ExampleEpFactory*>(this_ptr);
    return ep->ep_name_.c_str();
  }

  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) {
    const auto* ep = static_cast<const ExampleEpFactory*>(this_ptr);
    return ep->vendor_.c_str();
  }

  static bool GetDeviceInfoIfSupportedImpl(const OrtEpFactory* this_ptr,
                                           const OrtHardwareDevice* device,
                                           _Out_opt_ OrtKeyValuePairs** ep_metadata,
                                           _Out_opt_ OrtKeyValuePairs** ep_options) {
    const auto* ep = static_cast<const ExampleEpFactory*>(this_ptr);

    if (ep->ort_api.HardwareDevice_Type(device) == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      // these can be returned as nullptr if you have nothing to add.
      ep->ort_api.CreateKeyValuePairs(ep_metadata);
      ep->ort_api.CreateKeyValuePairs(ep_options);

      // random example using made up values
      ep->ort_api.AddKeyValuePair(*ep_metadata, "version", "0.1");
      ep->ort_api.AddKeyValuePair(*ep_options, "run_really_fast", "true");

      return true;
    }

    return nullptr;
  }

  static OrtStatus* CreateEpImpl(OrtEpFactory* this_ptr,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                 _In_ size_t num_devices,
                                 _In_ const OrtSessionOptions* session_options,
                                 _In_ const OrtLogger* logger,
                                 _Out_ OrtEp** ep) {
    auto* factory = static_cast<ExampleEpFactory*>(this_ptr);

    if (num_devices != 1) {
      // we only registered for CPU and only expected to be selected for one CPU
      // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
      // the EP has been selected for.
      return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                           "Example EP only supports selection for one device.");
    }

    // Create the execution provider
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                       OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                       "Creating Example EP", ORT_FILE, __LINE__, __FUNCTION__));

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

    auto dummy_ep = std::make_unique<ExampleEp>(*factory, factory->ep_name_, *session_options, *logger);

    *ep = dummy_ep.release();
    return nullptr;
  }

  static void ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) {
    ExampleEp* dummy_ep = static_cast<ExampleEp*>(ep);
    delete dummy_ep;
  }

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name
};

//
// Public symbols
//
OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                             OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<ExampleEpFactory>(registration_name,
                                                                             ApiPtrs{*ort_api, *ep_api});

  assert(max_factories > 1);  // we need at least one slot for the factory
  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete factory;
  return nullptr;
}
