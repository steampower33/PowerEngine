#pragma once

struct Vertex;
struct Camera;

#include "vulkan_utils.h"

class Model
{
public:
	Model(const std::string modelPath, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Device& device, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, uint32_t& model_count, glm::vec3 initPos);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	~Model() = default;

	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;

	struct UniformData {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	} uniform_data;

	void LoadModel(const std::string& modelPath);

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
	float radius_ = 1.0f;

	std::vector<Vertex> vertices_;
	std::vector<uint32_t> indices_;
	vk::raii::Buffer vertex_buffer_{ nullptr };
	vk::raii::DeviceMemory vertex_buffer_memory_{ nullptr };
	vk::raii::Buffer index_buffer_{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory_{ nullptr };

	void ApplyTransform(const glm::quat& rotationDelta, const glm::vec3& translationDelta);

};