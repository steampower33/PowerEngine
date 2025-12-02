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
		int env = -1;
		int radiance = -1;
		int irradiance = -1;
	} texture_idx_;

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
};