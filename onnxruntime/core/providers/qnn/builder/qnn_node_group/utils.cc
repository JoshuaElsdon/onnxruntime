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
                                    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit, const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group) {
  // the main node is the one that should have an input of the given type. For example it could be a MatMulNBits node that has a DequantizeLinear as input on its scale input (input index 2).
  const Node& main_node = main_node_unit.GetNode();
  // check that there are at least input_index inputs
  if (main_node.InputDefs().size() <= input_index) {
    return nullptr;
  }
  const NodeArg& input_node_arg = main_node.InputDefs()[input_index];
  const Node* input_node = graph_viewer.GetProducerNode(input_node_arg.Name());
  if (input_node == nullptr) {
    return nullptr;  // Node is not in this GraphViewer
  }

  // check if the input node is a valid type
  const std::string& input_type = input_node->OpType();
  bool is_valid_input_type = false;
  for (const auto& valid_op_type : input_types) {
    if (valid_op_type == input_type) {
      is_valid_input_type = true;
      break;
    }
  }
  if (!is_valid_input_type) {
    return nullptr;
  }

  const auto input_node_unit_it = node_unit_map.find(&input_node);
  if (input_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const NodeUnit* input_node_unit = input_node_unit_it->second;

  // Check if input node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (node_unit_to_qnn_node_group.count(input_node_unit) != 0) {
    return nullptr;
  }
  // input must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (input_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return input_node_unit;

}  // namespace qnn
}  // namespace onnxruntime
