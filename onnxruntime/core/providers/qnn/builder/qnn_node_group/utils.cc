#include "core/providers/qnn/builder/qnn_node_group/utils.h"

#include <gsl/gsl>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_node_group.h"

namespace onnxruntime {
namespace qnn {

const NodeUnit* GetOnlyChildOfType(const GraphViewer& graph_viewer,
                                   const NodeUnit& parent_node_unit,
                                   gsl::span<const std::string_view> child_op_types,
                                   const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                   const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Node& parent_node = parent_node_unit.GetNode();

  // Parent must have a single child (1 output edge) and must not produce a graph output.
  if (parent_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(parent_node)) {
    return nullptr;
  }

  // Child must be of a valid type.
  const Node& child_node = parent_node.OutputEdgesBegin()->GetNode();
  if (graph_viewer.GetNode(child_node.Index()) == nullptr) {
    return nullptr;  // Node is not in this GraphViewer
  }
  const std::string& child_type = child_node.OpType();
  bool is_valid_child_type = false;

  for (const auto& valid_op_type : child_op_types) {
    if (valid_op_type == child_type) {
      is_valid_child_type = true;
      break;
    }
  }

  if (!is_valid_child_type) {
    return nullptr;
  }

  const auto child_node_unit_it = node_unit_map.find(&child_node);
  if (child_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const NodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (qnn_node_group_map.count(child_node_unit) != 0) {
    return nullptr;
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return child_node_unit;
}

const NodeUnit* GetInputTypeOnIndex(const GraphViewer& graph_viewer, const NodeUnit& main_node_unit, const int input_index, gsl::span<const std::string_view> input_types,
                                    const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map, const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map, const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(node_unit_map);
  ORT_UNUSED_PARAMETER(qnn_node_group_map);
  ORT_UNUSED_PARAMETER(graph_viewer);
  // the main node is the one that should have an input of the given type. For example it could be a MatMulNBits node that has a DequantizeLinear as input on its scale input (input index 2).
  const Node& main_node = main_node_unit.GetNode();
  // input_count is the length of the possible inputs, holding the integer of the numper of times that input is used (should be max of 1)
  const std::vector<int>& input_count = main_node.InputArgCount();

  LOGS(logger, INFO) << "GetInputTypeOnIndex() main_node: " << main_node.Name() << " input_index: " << input_index << " input_types: " << input_types[0] << " input_count: " << input_count.size();

  // get the input_defs
  ConstPointerContainer<std::vector<NodeArg*>> input_defs = main_node.InputDefs();

  for (size_t i = 0; i < input_defs.size(); ++i) {
    LOGS(logger, INFO) << "GetInputTypeOnIndex() input_defs[" << i << "]: " << input_defs[i]->Name();
  }

  // find the input of interest.
  const std::string input_of_intersest = input_defs[input_index]->Name();
  LOGS(logger, INFO) << "GetInputTypeOnIndex() input_of_intersest: " << input_of_intersest;

  // find a node that produces the input of interest.
  auto input_node_it = main_node.InputNodesBegin();
  for (; input_node_it != main_node.InputNodesEnd(); ++input_node_it) {
    // check if output of the input node is the input of interest
    if (input_node_it->OutputDefs()[0]->Name() == input_of_intersest) {
      break;
    }
  }
  if (!(input_node_it != main_node.InputNodesEnd())) {
    LOGS(logger, INFO) << "GetInputTypeOnIndex() could not find input_node for input_of_intersest: " << input_of_intersest;
    return nullptr;
  }

  const std::string& input_node_type = input_node_it->OpType();

  // check if the input node is a valid type
  bool is_valid_input_type = false;
  for (const auto& valid_op_type : input_types) {
    if (valid_op_type == input_node_type) {
      is_valid_input_type = true;
      break;
    }
  }
  if (!is_valid_input_type) {
    LOGS(logger, INFO) << "GetInputTypeOnIndex() input_node: " << input_node_it->Name() << " is not a valid type: " << input_node_type;
    return nullptr;
  }

  const auto input_node_unit_it = node_unit_map.find(&*input_node_it);
  if (input_node_unit_it == node_unit_map.end()) {
    LOGS(logger, INFO) << "GetInputTypeOnIndex() could not find input_node: " << input_node_it->Name() << " in node_unit_map";
    return nullptr;
  }
  const NodeUnit* input_node_unit = input_node_unit_it->second;

  // Check if input node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (qnn_node_group_map.count(input_node_unit) != 0) {
    LOGS(logger, INFO) << "GetInputTypeOnIndex() input_node: " << input_node_it->Name() << " already handled";
    return nullptr;
  }
  // input must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (input_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    LOGS(logger, INFO) << "GetInputTypeOnIndex() input_node: " << input_node_it->Name() << " is not a standalone node";
    return nullptr;
  }
  // LOGS(logger, INFO) << "HHHHAAAACKKK";
  // return nullptr;
  return input_node_unit;
}

}  // namespace qnn
}  // namespace onnxruntime
