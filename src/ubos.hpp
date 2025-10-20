
struct ModelUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct ParticleUBO {
    float deltaTime = 1.0f;
};