// src/torchscience/csrc/autograd/space_partitioning/octree.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::space_partitioning {

class OctreeSample : public torch::autograd::Function<OctreeSample> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& data,
        const at::Tensor& codes,
        const at::Tensor& structure,
        const at::Tensor& children_mask,
        const at::Tensor& points,
        int64_t maximum_depth,
        int64_t interpolation,
        c10::optional<int64_t> query_depth
    ) {
        ctx->saved_data["maximum_depth"] = maximum_depth;
        ctx->saved_data["interpolation"] = interpolation;
        ctx->saved_data["query_depth"] = query_depth.has_value() ?
            c10::IValue(query_depth.value()) : c10::IValue();
        ctx->saved_data["data_requires_grad"] = data.requires_grad();
        ctx->saved_data["points_requires_grad"] = points.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [output_data, found] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::octree_sample", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&,
                int64_t, int64_t, c10::optional<int64_t>
            )>()
            .call(data, codes, structure, children_mask, points,
                  maximum_depth, interpolation, query_depth);

        ctx->save_for_backward({data, codes, structure, children_mask, points});

        return {output_data, found};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor data = saved[0];
        at::Tensor codes = saved[1];
        at::Tensor structure = saved[2];
        at::Tensor children_mask = saved[3];
        at::Tensor points = saved[4];

        int64_t maximum_depth = ctx->saved_data["maximum_depth"].toInt();
        int64_t interpolation = ctx->saved_data["interpolation"].toInt();
        c10::optional<int64_t> query_depth;
        if (!ctx->saved_data["query_depth"].isNone()) {
            query_depth = ctx->saved_data["query_depth"].toInt();
        }
        bool data_requires_grad = ctx->saved_data["data_requires_grad"].toBool();
        bool points_requires_grad = ctx->saved_data["points_requires_grad"].toBool();

        at::Tensor grad_output = grad_outputs[0];  // grad for output_data
        // grad_outputs[1] is grad for found (bool tensor, no gradient)

        if (!data_requires_grad && !points_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_data, grad_points] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::octree_sample_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                int64_t, int64_t, c10::optional<int64_t>
            )>()
            .call(grad_output, data, codes, structure, children_mask, points,
                  maximum_depth, interpolation, query_depth);

        // Return None for non-differentiable inputs
        return {
            data_requires_grad ? grad_data : at::Tensor(),
            at::Tensor(),  // codes (structure, no gradient)
            at::Tensor(),  // structure (no gradient)
            at::Tensor(),  // children_mask (no gradient)
            points_requires_grad ? grad_points : at::Tensor(),
            at::Tensor(),  // maximum_depth (scalar, no gradient)
            at::Tensor(),  // interpolation (scalar, no gradient)
            at::Tensor()   // query_depth (scalar, no gradient)
        };
    }
};

inline std::tuple<at::Tensor, at::Tensor> octree_sample(
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& points,
    int64_t maximum_depth,
    int64_t interpolation,
    c10::optional<int64_t> query_depth
) {
    auto results = OctreeSample::apply(
        data, codes, structure, children_mask, points,
        maximum_depth, interpolation, query_depth
    );
    return std::make_tuple(results[0], results[1]);
}

class OctreeRayMarching : public torch::autograd::Function<OctreeRayMarching> {
public:
    static std::vector<at::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& data,
        const at::Tensor& codes,
        const at::Tensor& structure,
        const at::Tensor& children_mask,
        const at::Tensor& origins,
        const at::Tensor& directions,
        int64_t maximum_depth,
        c10::optional<double> step_size,
        int64_t maximum_steps
    ) {
        ctx->saved_data["maximum_depth"] = maximum_depth;
        ctx->saved_data["step_size"] = step_size.has_value() ?
            c10::IValue(step_size.value()) : c10::IValue();
        ctx->saved_data["maximum_steps"] = maximum_steps;
        ctx->saved_data["data_requires_grad"] = data.requires_grad();
        ctx->saved_data["origins_requires_grad"] = origins.requires_grad();
        ctx->saved_data["directions_requires_grad"] = directions.requires_grad();

        at::AutoDispatchBelowAutograd guard;

        auto [positions, out_data, mask] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::octree_ray_marching", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                int64_t, c10::optional<double>, int64_t
            )>()
            .call(data, codes, structure, children_mask, origins, directions,
                  maximum_depth, step_size, maximum_steps);

        ctx->save_for_backward({data, codes, structure, children_mask, origins, directions, mask});

        return {positions, out_data, mask};
    }

    static std::vector<at::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        const std::vector<at::Tensor>& grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        at::Tensor data = saved[0];
        at::Tensor codes = saved[1];
        at::Tensor structure = saved[2];
        at::Tensor children_mask = saved[3];
        at::Tensor origins = saved[4];
        at::Tensor directions = saved[5];
        at::Tensor mask = saved[6];

        int64_t maximum_depth = ctx->saved_data["maximum_depth"].toInt();
        c10::optional<double> step_size;
        if (!ctx->saved_data["step_size"].isNone()) {
            step_size = ctx->saved_data["step_size"].toDouble();
        }
        int64_t maximum_steps = ctx->saved_data["maximum_steps"].toInt();
        bool data_requires_grad = ctx->saved_data["data_requires_grad"].toBool();
        bool origins_requires_grad = ctx->saved_data["origins_requires_grad"].toBool();
        bool directions_requires_grad = ctx->saved_data["directions_requires_grad"].toBool();

        at::Tensor grad_positions = grad_outputs[0];  // grad for positions
        at::Tensor grad_data_out = grad_outputs[1];   // grad for out_data
        // grad_outputs[2] is grad for mask (bool tensor, no gradient)

        if (!data_requires_grad && !origins_requires_grad && !directions_requires_grad) {
            return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        at::AutoDispatchBelowAutograd guard;

        auto [grad_data, grad_origins, grad_directions] = c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::octree_ray_marching_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                const at::Tensor&, const at::Tensor&, const at::Tensor&,
                int64_t, c10::optional<double>, int64_t
            )>()
            .call(grad_positions, grad_data_out, mask, data, codes, structure,
                  children_mask, origins, directions,
                  maximum_depth, step_size, maximum_steps);

        return {
            data_requires_grad ? grad_data : at::Tensor(),
            at::Tensor(),  // codes (structure, no gradient)
            at::Tensor(),  // structure (no gradient)
            at::Tensor(),  // children_mask (no gradient)
            origins_requires_grad ? grad_origins : at::Tensor(),
            directions_requires_grad ? grad_directions : at::Tensor(),
            at::Tensor(),  // maximum_depth (scalar, no gradient)
            at::Tensor(),  // step_size (scalar, no gradient)
            at::Tensor()   // maximum_steps (scalar, no gradient)
        };
    }
};

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> octree_ray_marching(
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& origins,
    const at::Tensor& directions,
    int64_t maximum_depth,
    c10::optional<double> step_size,
    int64_t maximum_steps
) {
    auto results = OctreeRayMarching::apply(
        data, codes, structure, children_mask, origins, directions,
        maximum_depth, step_size, maximum_steps
    );
    return std::make_tuple(results[0], results[1], results[2]);
}

}  // namespace torchscience::autograd::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
    m.impl("octree_sample", &torchscience::autograd::space_partitioning::octree_sample);
    m.impl("octree_ray_marching", &torchscience::autograd::space_partitioning::octree_ray_marching);
}
