#pragma once

#include "vulkan_utils.h"

// 파이프라인 만들 때 쓰기 편하게 묶어둔 구조체
struct VertexInputDescription {
    std::vector<vk::VertexInputBindingDescription>   bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;
};

struct Vertex {
    glm::vec3 pos = {};
    glm::vec2 uv = {};
    glm::vec3 normal = {};
    glm::vec3 tangent = {};

    // 1) 바인딩 + 어트리뷰트 자동 생성기
    static VertexInputDescription GetInputDescription(const vku::VertexIncludeInfo& include)
    {
        VertexInputDescription desc;

        // 하나의 바인딩(0번)만 사용
        vk::VertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = vk::VertexInputRate::eVertex;

        desc.bindings.push_back(binding);

        // location 0 : position (vec3)
        vk::VertexInputAttributeDescription posAttr{};
        posAttr.location = 0;
        posAttr.binding = 0;
        posAttr.format = vk::Format::eR32G32B32Sfloat;
        posAttr.offset = static_cast<uint32_t>(offsetof(Vertex, pos));
        desc.attributes.push_back(posAttr);
        
        if (include.uv)
        {
            // location 1 : uv (vec2)
            vk::VertexInputAttributeDescription uvAttr{};
            uvAttr.location = 1;
            uvAttr.binding = 0;
            uvAttr.format = vk::Format::eR32G32Sfloat;
            uvAttr.offset = static_cast<uint32_t>(offsetof(Vertex, uv));
            desc.attributes.push_back(uvAttr);
        }

        // 나머지 location은 include에 따라 선택적으로 추가
        // (원하면 항상 추가해서 고정 레이아웃으로 써도 됨)
        if (include.normal) {
            vk::VertexInputAttributeDescription nAttr{};
            nAttr.location = 2;
            nAttr.binding = 0;
            nAttr.format = vk::Format::eR32G32B32Sfloat;
            nAttr.offset = static_cast<uint32_t>(offsetof(Vertex, normal));
            desc.attributes.push_back(nAttr);
        }

        if (include.tangent) {
            vk::VertexInputAttributeDescription tAttr{};
            tAttr.location = 3;
            tAttr.binding = 0;
            tAttr.format = vk::Format::eR32G32B32A32Sfloat;
            tAttr.offset = static_cast<uint32_t>(offsetof(Vertex, tangent));
            desc.attributes.push_back(tAttr);
        }

        return desc;
    }
};