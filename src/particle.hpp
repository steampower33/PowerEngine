struct Particle {
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Particle), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Particle, color)),
        };
    }
};