#include "model.h"
#include "context.h"
#include "graphics_context.h"
#include "geometry_generator.h"
#include "texture_manager.h"

#include "model_manager.h"

ModelManager::ModelManager(Context& context, GraphicsContext& graphicsContext, TextureManager& textureManager)
{
	models.reserve(kMaxObjects);

	{
		//MeshData sphere = GeometryGenerator::MakeBox(1.0f);
		//MeshData sphere = GeometryGenerator::MakeCylinder(1.0f, 1.0f, 5.0f, 4);
		MeshData sphere = GeometryGenerator::MakeSphere(1.0f, 20, 20);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(244.0f / 255.0f, 114 / 255.0f, 43 / 255.0f, 0.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(sphere, vku::VertexIncludeInfo{ true, true }, context, graphicsContext, model_count_, initPos, angleQuat, initColor, true);
		model->texture_idx_.albedo = textureManager.CreateTexture2D("assets/Metal", "albedo");
		model->texture_idx_.metallic = textureManager.CreateTexture2D("assets/Metal", "metallic");
		model->texture_idx_.normal = textureManager.CreateTexture2D("assets/Metal", "normal");
		model->texture_idx_.roughness = textureManager.CreateTexture2D("assets/Metal", "roughness");
		model->texture_idx_.ao = textureManager.CreateTexture2D("assets/Metal", "ao");
		model->texture_idx_.height = textureManager.CreateTexture2D("assets/Metal", "height");
		models.emplace_back(std::move(model));
	}

	{
		MeshData plane = GeometryGenerator::MakeSquare(10.0f);
		glm::quat angleQuat = glm::angleAxis(glm::radians(90.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 0.0f, 0.0f);
		float color = 62.0f / 255.0f;
		glm::vec4 initColor = glm::vec4(color, color, color, 1.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(plane, vku::VertexIncludeInfo{ true, true }, context, graphicsContext, model_count_, initPos, angleQuat, initColor, false);
		models.emplace_back(std::move(model));
	}
}

ModelManager::~ModelManager()
{

}