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
		float radius = 0.5f;
		MeshData sphere = GeometryGenerator::MakeSphere(radius, 20, 20);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(sphere, vku::VertexIncludeInfo{ true, true }, context, graphicsContext, model_count_, initPos, angleQuat, initColor, true);
		model->radius_ = radius;
		//model->texture_idx_.albedo = textureManager.CreateTexture("assets/Metal", "albedo");
		//model->texture_idx_.metallic = textureManager.CreateTexture("assets/Metal", "metallic");
		//model->texture_idx_.normal = textureManager.CreateTexture("assets/Metal", "normal");
		//model->texture_idx_.roughness = textureManager.CreateTexture("assets/Metal", "roughness");
		//model->texture_idx_.ao = textureManager.CreateTexture("assets/Metal", "ao");
		//model->texture_idx_.height = textureManager.CreateTexture("assets/Metal", "height");
		models.emplace_back(std::move(model));
	}

	{
		//MeshData plane = GeometryGenerator::MakeSquare(10.0f);
		//glm::quat angleQuat = glm::angleAxis(glm::radians(90.0f), glm::vec3(1, 0, 0));
		//glm::vec3 initPos = glm::vec3(0.0f, 0.0f, 0.0f);
		//float color = 62.0f / 255.0f;
		//glm::vec4 initColor = glm::vec4(color, color, color, 1.0);
		//std::unique_ptr<Model> model = std::make_unique<Model>(plane, vku::VertexIncludeInfo{ true, true }, context, graphicsContext, model_count_, initPos, angleQuat, initColor, false);
		//model->texture_idx_.albedo = textureManager.CreateTexture("assets/Fabric", "color");
		//model->texture_idx_.metallic = 0;
		//model->texture_idx_.normal = textureManager.CreateTexture("assets/Fabric", "normalgl");
		//model->texture_idx_.roughness = textureManager.CreateTexture("assets/Fabric", "roughness");
		//model->texture_idx_.ao = 0;
		//model->texture_idx_.height = textureManager.CreateTexture("assets/Fabric", "displacement");
		//models.emplace_back(std::move(model));
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