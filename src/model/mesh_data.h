#pragma once

#include "vertex.h"

struct MeshData
{
	std::vector<Vertex> vertices;
	vk::raii::Buffer vertex_buffer{ nullptr };
	vk::raii::DeviceMemory vertex_buffer_memory{ nullptr };

	std::vector<uint32_t> indices;
	vk::raii::Buffer index_buffer{ nullptr };
	vk::raii::DeviceMemory index_buffer_memory{ nullptr };
	uint32_t indices_count;

	std::vector<glm::uvec4> tets;
};