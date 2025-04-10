// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
// struct to provider ownership via std::string as well as support the GetKeyValuePairs
// TODO: Validate adding entries doesn't invalidate existing pointers. assuming std::unordered_map is smart enough to
//       std::move any strings in it.

struct OrtKeyValuePairs {
  std::unordered_map<std::string, std::string> entries;
  // members to make returning all key/value entries via the C API easier
  std::vector<const char*> keys;
  std::vector<const char*> values;

  void Copy(const std::unordered_map<std::string, std::string>& src) {
    entries = src;
    Sync();
  }

  void Add(const char* key, const char* value) {
    std::string key_str(key);
    auto iter_inserted = entries.insert({std::move(key_str), std::string(value)});
    bool inserted = iter_inserted.second;
    if (inserted) {
      const auto& entry = *iter_inserted.first;
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    } else {
      // rebuild is easier and this is not expected to be a common case. otherwise we need to to strcmp on all entries.
      Sync();
    }
  }

  // we don't expect this to be common. reconsider using std::vector or call Sync if it turns out to be.
  void Remove(const char* key) {
    auto iter = entries.find(key);
    if (iter != entries.end()) {
      auto key_iter = std::find(keys.begin(), keys.end(), iter->first.c_str());
      // there should only ever be one matching entry, and keys and values should be in sync
      if (key_iter != keys.end()) {
        auto idx = std::distance(keys.begin(), key_iter);
        keys.erase(key_iter);
        values.erase(values.begin() + idx);
      }
    }
  }

 private:
  void Sync() {
    keys.clear();
    values.clear();
    for (const auto& entry : entries) {
      keys.push_back(entry.first.c_str());
      values.push_back(entry.second.c_str());
    }
  }
};
