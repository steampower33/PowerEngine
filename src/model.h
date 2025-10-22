#pragma once

struct Vertex;

class Model
{
public:
	Model(const std::string modelPath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	~Model() = default;

	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;

	void LoadModel(const std::string& modelPath);

	glm::vec3 position_ = { 0.0f, 0.0f, 0.0f };
	glm::vec3 rotation_ = { 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_ = { 1.0f, 1.0f, 1.0f };

	std::vector<Vertex> vertices_;
	std::vector<uint32_t> indices_;
	vk::raii::Buffer vertex_buffer_ = nullptr;
	vk::raii::DeviceMemory vertex_buffer_memory_ = nullptr;
	vk::raii::Buffer index_buffer_ = nullptr;
	vk::raii::DeviceMemory index_buffer_memory_ = nullptr;
};