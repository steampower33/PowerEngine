#pragma once

class Model;

struct BoundingSphere
{
    glm::vec3 center{ 0.0f };
    float radius = 0.0f;
};

class Ray
{
public:
    Ray(const glm::vec3& origin, const glm::vec3& direction);

    bool Intersects(const Model& model, float& dist) const;

public:
    glm::vec3 origin;
    glm::vec3 direction;
};