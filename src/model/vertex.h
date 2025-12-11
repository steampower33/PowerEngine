#pragma once

#include "vulkan_utils.h"

struct VertexInputDescription {
    std::vector<vk::VertexInputBindingDescription>   bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
};

struct Vertex {
    glm::vec3 pos = {};
    glm::vec2 uv = {};
    glm::vec3 normal = {};
    glm::vec3 tangent = {};
    glm::uvec4 joints = {};   // 4 bone indices
    glm::vec4 weights = {};   // 4 weights

    static VertexInputDescription GetInputDescription(const vku::VertexIncludeInfo& include)
    {
        VertexInputDescription desc;

        vk::VertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = vk::VertexInputRate::eVertex;

        desc.bindings.push_back(binding);

        uint32_t location = 0;

        // location 0 : position (vec3)
        vk::VertexInputAttributeDescription posAttr{};
        posAttr.location = location++;
        posAttr.binding = 0;
        posAttr.format = vk::Format::eR32G32B32Sfloat;
        posAttr.offset = static_cast<uint32_t>(offsetof(Vertex, pos));
        desc.attributes.push_back(posAttr);
        
        if (include.uv)
        {
            // location 1 : uv (vec2)
            vk::VertexInputAttributeDescription uvAttr{};
            uvAttr.location = location++;
            uvAttr.binding = 0;
            uvAttr.format = vk::Format::eR32G32Sfloat;
            uvAttr.offset = static_cast<uint32_t>(offsetof(Vertex, uv));
            desc.attributes.push_back(uvAttr);
        }

        if (include.normal) {
            vk::VertexInputAttributeDescription nAttr{};
            nAttr.location = location++;
            nAttr.binding = 0;
            nAttr.format = vk::Format::eR32G32B32Sfloat;
            nAttr.offset = static_cast<uint32_t>(offsetof(Vertex, normal));
            desc.attributes.push_back(nAttr);
        }

        if (include.tangent) {
            vk::VertexInputAttributeDescription tAttr{};
            tAttr.location = location++;
            tAttr.binding = 0;
            tAttr.format = vk::Format::eR32G32B32Sfloat;
            tAttr.offset = static_cast<uint32_t>(offsetof(Vertex, tangent));
            desc.attributes.push_back(tAttr);
        }

        if (include.joints)
        {
            vk::VertexInputAttributeDescription attr{};
            attr.location = location++;
            attr.binding = 0;
            attr.format = vk::Format::eR32G32B32A32Uint;
            attr.offset = static_cast<uint32_t>(offsetof(Vertex, joints));
            desc.attributes.push_back(attr);
        }

        if (include.weights)
        {
            vk::VertexInputAttributeDescription attr{};
            attr.location = location++;
            attr.binding = 0;
            attr.format = vk::Format::eR32G32B32A32Sfloat;
            attr.offset = static_cast<uint32_t>(offsetof(Vertex, weights));
            desc.attributes.push_back(attr);
        }

        return desc;
    }
};