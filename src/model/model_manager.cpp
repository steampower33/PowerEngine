#include "model.h"
#include "context.h"
#include "pass_manager.h"
#include "geometry_generator.h"
#include "texture_manager.h"
#include "skybox.h"

#include "model_manager.h"

ModelManager::ModelManager(Context& context, TextureManager& textureManager)
{
	models_.reserve(kMaxModels);

	{
		//MeshData sphere = GeometryGenerator::MakeBox(1.0f);
		//MeshData sphere = GeometryGenerator::MakeCylinder(1.0f, 1.0f, 5.0f, 4);
		float radius = 0.25f;
		Mesh sphere = GeometryGenerator::MakeSphere(radius, 20, 20);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(sphere, vku::VertexIncludeInfo{ true, true, true, false, false }, context, initPos, angleQuat, initColor, true, "Sphere");
		model->radius_ = radius;
		model->type_ = Model::Type::NORMAL;
		models_.emplace_back(std::move(model));
	}

	{
		Mesh mesh = GeometryGenerator::MakeSquare(1000.0f);
		glm::quat angleQuat = glm::angleAxis(glm::radians(90.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		std::unique_ptr<Model> model = std::make_unique<Model>(mesh, vku::VertexIncludeInfo{ true, true, true, false, false }, context, initPos, angleQuat, initColor, false, "BottomPlane");
		model->factors_.roughness = 1.0f;
		model->factors_.metallic = 0.0f;
		model->checker_board_enable_ = true;
		model->type_ = Model::Type::NORMAL;

		models_.emplace_back(std::move(model));
	}

	{
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(-1.0f, 0.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);

		std::string filename = "assets/walking.glb";
		std::unique_ptr<Model> model = std::make_unique<Model>(filename, vku::VertexIncludeInfo{ true, true, true, true, true }, context, textureManager, initPos, angleQuat, initColor, true, "walking", 1.0f);
		model->type_ = Model::Type::SKINNED;

		models_.push_back(std::move(model));
	}

	{
		Mesh capsule = GeometryGenerator::MakeCapsule(0.5f, 0.5f, 0.5f, 16);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 1.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		debug_capsule_ = std::make_unique<Model>(capsule, vku::VertexIncludeInfo{ true, true, true, false, false }, context, initPos, angleQuat, initColor, true, "Capsule");
		debug_capsule_->type_ = Model::Type::NORMAL;
		//models_.emplace_back(std::move(model));
	}

	//{
	//	glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
	//	glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 1.0f);
	//	glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);

	//	//MeshData box = GeometryGenerator::MakeTetraBox(0.2f);
	//	//soft_body_ = std::make_unique<Model>(box, vku::VertexIncludeInfo{ true, true }, context, initPos, angleQuat, initColor, true, "soft_body_");

	//	std::string filename = "assets/SheenCloth/SheenCloth.gltf";
	//	std::unique_ptr<Model> model = std::make_unique<Model>(filename, vku::VertexIncludeInfo{true, true}, context, textureManager, initPos, angleQuat, initColor, true, "SheenCloth", 10.0f);
	//	models_.emplace_back(std::move(model));
	//}

	//{
	//	MeshData box = GeometryGenerator::MakeBox(50.0f);

	//	std::reverse(box.indices.begin(), box.indices.end());
	//	skybox_ = std::make_unique<Skybox>(box, vku::VertexIncludeInfo{ false, false }, context, passManager);

	//	skybox_->texture_idx_.env = textureManager.CreateTexture("assets/DaySky", "env", true);
	//	skybox_->texture_idx_.radiance = textureManager.CreateTexture("assets/DaySky", "specular", true);
	//	skybox_->texture_idx_.irradiance = textureManager.CreateTexture("assets/DaySky", "diffuse", true);
	//}
}

ModelManager::~ModelManager()
{

}