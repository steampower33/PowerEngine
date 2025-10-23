#pragma once

struct Vertex {
    glm::vec3 pos;
    glm::vec2 texcoord;
    //glm::vec3 normal;

    static vk::VertexInputBindingDescription GetBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> GetAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texcoord)),
            //vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal))
        };
    }

    bool operator==(const Vertex& other) const {
        //return pos == other.pos && normal == other.normal && texcoord == other.texcoord;
        return pos == other.pos && texcoord == other.texcoord;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            // pos, texcoord, normal을 모두 조합하여 해시값 생성
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec2>()(vertex.texcoord) << 1)) >> 1);
            //^ (hash<glm::vec3>()(vertex.normal) << 1);
        }
    };
}
