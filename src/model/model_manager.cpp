#include "model.h"
#include "context.h"
#include "graphics_context.h"
#include "geometry_generator.h"
#include "texture_manager.h"
#include "skybox.h"

#include "model_manager.h"

ModelManager::ModelManager(Context& context, GraphicsContext& graphicsContext, TextureManager& textureManager)
{
	models.reserve(kMaxObjects);

	{
		//MeshData sphere = GeometryGenerator::MakeBox(1.0f);
		//MeshData sphere = GeometryGenerator::MakeCylinder(1.0f, 1.0f, 5.0f, 4);
		float radius = 0.125;
		MeshData sphere = GeometryGenerator::MakeSphere(radius, 20, 20);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(sphere, vku::VertexIncludeInfo{ true, true }, context, graphicsContext, initPos, angleQuat, initColor, true);
		model->radius_ = radius;
		models.emplace_back(std::move(model));
	}

	//{
	//	glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
	//	glm::vec3 initPos = glm::vec3(3.0f, 1.0f, 0.0f);
	//	glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
	//	std::string modelPath = "assets/ybot/ybot.gltf";
	//	std::unique_ptr<Model> model = std::make_unique<Model>(modelPath, vku::VertexIncludeInfo{true, true}, context, graphicsContext, textureManager, initPos, angleQuat, initColor, true);
	//	models.emplace_back(std::move(model));
	//}

	{
		MeshData sphere = GeometryGenerator::MakeSquare(1.0f);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(sphere, vku::VertexIncludeInfo{ true, true }, context, graphicsContext, initPos, angleQuat, initColor, true);

		model->texture_idx_.albedo = textureManager.CreateTexture("assets/Fabric", "color");
		model->texture_use_.albedo = (model->texture_idx_.albedo != -1) ? 1 : 0;
		model->texture_idx_.normal = textureManager.CreateTexture("assets/Fabric", "normalgl");
		model->texture_use_.normal = (model->texture_idx_.normal != -1) ? 1 : 0;
		model->texture_idx_.height = textureManager.CreateTexture("assets/Fabric", "displacement");
		model->texture_use_.height = (model->texture_idx_.height != -1) ? 1 : 0;
		model->texture_idx_.roughness = textureManager.CreateTexture("assets/Fabric", "roughness");
		model->texture_use_.roughtness = (model->texture_idx_.roughness != -1) ? 1 : 0;

		cloth_ = std::move(model);
	}

	{
		MeshData box = GeometryGenerator::MakeBox(50.0f);

		std::reverse(box.indices.begin(), box.indices.end());
		skybox_ = std::make_unique<Skybox>(box, vku::VertexIncludeInfo{ false, false }, context, graphicsContext);

		skybox_->texture_idx_.env = textureManager.CreateTexture("assets/Indoor", "env", true);
		skybox_->texture_idx_.radiance = textureManager.CreateTexture("assets/Indoor", "radiance_specular", true);
		skybox_->texture_idx_.irradiance = textureManager.CreateTexture("assets/Indoor", "irradiance_diffuse", true);
	}
}

ModelManager::~ModelManager()
{

}