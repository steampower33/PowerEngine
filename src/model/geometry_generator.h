#pragma once

#include "model_data.h"

class GeometryGenerator {
public:

	static Mesh MakeSquare(const float scale = 1.0f);
	static Mesh MakeBox(const float scale = 1.0f);
	static Mesh MakeCylinder(
		const float bottomRadius,
		const float topRadius, float height,
		int sliceCount);
	static Mesh MakeCapsule(
		const float bottomRadius,
		const float topRadius, float height,
		int sliceCount);
	static Mesh MakeSphere(
		const float radius,
		const int numSlices, const int numStacks, const glm::vec2 texScale = { 1.0f, 1.0f });
	static void CalculateTangents(Mesh& meshData);

};