#pragma once

class Context;
class GraphicsContext;
struct Vertex;
struct Camera;

#include "vulkan_utils.h"
#include "mesh_data.h"

class Model
{
public:
	Model(const std::string& modelPath, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, uint32_t& model_count, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble);
	Model(MeshData& meshData, vku::VertexIncludeInfo vertexIncludeInfo, Context& context, GraphicsContext& graphicsContext, uint32_t& model_count, glm::vec3 initPos, glm::quat initRotation, glm::vec4 colorUse, bool moveble);
	Model(const Model& rhs) = delete;
	Model(Model&& rhs) = delete;
	~Model() = default;

	Model& operator=(const Model& rhs) = delete;
	Model& operator=(Model&& rhs) = delete;

	glm::vec4 color_use{ 0.0f };

	void LoadModel(const std::string& modelPath, const vku::VertexIncludeInfo& vertexIncludeInfo);

	glm::mat4 world_{ 1.0f };
	glm::vec3 position_{ 0.0f, 0.0f, 0.0f };
	glm::quat rotation_{ 1.0f, 0.0f, 0.0f, 0.0f };
	glm::vec3 scale_{ 1.0f, 1.0f, 1.0f };
	float radius_ = 1.0f;

	bool moveble_ = false;

	MeshData mesh_data_;

	void ApplyTransform(const glm::quat& rotationDelta, const glm::vec3& translationDelta);

};