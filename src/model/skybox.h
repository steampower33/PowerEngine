#pragma once

class Context;
class GraphicsContext;
struct Vertex;

#include "vulkan_utils.h"
#include "mesh_data.h"

struct Skybox {
public:
	Skybox(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext);
	Skybox(const Skybox& rhs) = delete;
	Skybox(Skybox&& rhs) = delete;
	Skybox& operator=(const Skybox& rhs) = delete;
	Skybox& operator=(Skybox&& rhs) = delete;
	~Skybox() = default;

	MeshData mesh_data_;

	struct TextureIdx {
		uint32_t env = 0;
		uint32_t radiance = 0;
		uint32_t irradiance = 0;
	} texture_idx_;

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
};