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
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(-1.0f, 1.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		float initRadius = 0.25f;
		Mesh sphere = GeometryGenerator::MakeSphere(initRadius, 20, 20);

		std::unique_ptr<Model> model = std::make_unique<Model>(sphere, vku::VertexIncludeInfo{ true, true, true, false, false }, context, initPos, angleQuat, initColor, initRadius, true, "Sphere", ShapeColliderType::SPHERE, ModelType::SHAPE);
		models_.emplace_back(std::move(model));
	}

	{
		Mesh mesh = GeometryGenerator::MakeSquare(1000.0f);
		glm::quat angleQuat = glm::angleAxis(glm::radians(90.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		float initRadius = 1.0f;
		std::unique_ptr<Model> model = std::make_unique<Model>(mesh, vku::VertexIncludeInfo{ true, true, true, false, false }, context, initPos, angleQuat, initColor, initRadius, false, "BottomPlane", ShapeColliderType::PLANE, ModelType::SHAPE);
		model->factors_.roughness = 1.0f;
		model->factors_.metallic = 0.0f;
		model->checker_board_enable_ = true;

		models_.emplace_back(std::move(model));
	}

	{
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 2.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		float initRadius = 1.0f;

		std::string filename = "assets/cloth.glb";
		std::unique_ptr<Model> model = std::make_unique<Model>(filename, vku::VertexIncludeInfo{ true, true, true, false, false }, context, textureManager, initPos, angleQuat, initColor, initRadius, true, "Cloth Model", 1.0f, ShapeColliderType::NONE, ModelType::SHAPE);

		cloth_ = std::move(model);
		//models_.push_back(std::move(model));
	}

	{
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 0.0f, 0.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		float initRadius = 1.0f;

		std::string filename = "assets/walking.glb";
		std::unique_ptr<Model> model = std::make_unique<Model>(filename, vku::VertexIncludeInfo{ true, true, true, true, true }, context, textureManager, initPos, angleQuat, initColor, initRadius, true, "Walking", 1.0f, ShapeColliderType::NONE, ModelType::SKINNED);
		model->movable_ = false;

		models_.push_back(std::move(model));
	}

	{
		Mesh capsule = GeometryGenerator::MakeCapsule(0.5f, 0.5f, 0.5f, 16);
		glm::quat angleQuat = glm::angleAxis(glm::radians(0.0f), glm::vec3(1, 0, 0));
		glm::vec3 initPos = glm::vec3(0.0f, 1.0f, 1.0f);
		glm::vec4 initColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0);
		float initRadius = 1.0f;
		debug_capsule_ = std::make_unique<Model>(capsule, vku::VertexIncludeInfo{ true, true, true, false, false }, context, initPos, angleQuat, initColor, initRadius, true, "DebugCapsule", ShapeColliderType::CAPSULE, ModelType::SHAPE);
		debug_capsule_->render_ = false;

		//models_.emplace_back(std::move(model));
	}

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

void ModelManager::Update()
{

	// Update Animation and CapsuleColliders
	for (uint32_t i = 0; i < models_.size(); i++)
	{
		auto& model = *models_[i];

		if (model.model_type_ == ModelType::SKINNED)
		{
			/*model.UpdateSkinMatrices();*/
			model.current_time_ += 1.0f / 240.0f;
			model.ApplyAnimation(0, model.current_time_);
			model.UpdateCapsuleCollidersFromBones();

			auto& jm = model.skin_[0].jointMatrices;       // std::vector<glm::mat4>

			model.capsule_collision_update_ = true;
		}
	}
}