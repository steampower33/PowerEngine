#pragma once

#include "vertex.h"

struct Mesh
{
	std::vector<Vertex> vertices;
	vk::raii::Buffer vertex_buffer{ nullptr };
	vk::raii::DeviceMemory vertex_buffer_memory{ nullptr };

	std::vector<uint32_t> indices;
	vk::raii::Buffer index_buffer{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory{ nullptr };
	uint32_t indices_count;
};

struct Node
{
	int index = -1;
	int parent = -1;

	std::vector<int> children;

	glm::vec3 translation = glm::vec3(0.0f);
	glm::quat rotation = glm::quat(1, 0, 0, 0);   // identity
	glm::vec3 scale = glm::vec3(1.0f);

	glm::mat4 localMatrix = glm::mat4(1.0f);
	glm::mat4 worldMatrix = glm::mat4(1.0f);

	int meshIndex = -1;
	int skinIndex = -1;
};

struct Skin
{
	std::string name;
	std::vector<int> joints;         // glTF node indices
	std::vector<glm::mat4> inverseBindMatrices;
	int skeletonRoot = -1;
};