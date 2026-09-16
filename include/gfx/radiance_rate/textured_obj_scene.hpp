#pragma once

#include <filesystem>
#include <vector>

#include <glm/vec3.hpp>

#include <gfx/research/mesh.hpp>
#include <gfx/research/shader.hpp>
#include <gfx/research/texture_2d.hpp>

namespace gfx::radiance_rate {
    class TexturedObjScene {
    public:
        TexturedObjScene() = default;
        TexturedObjScene(const TexturedObjScene&) = delete;
        TexturedObjScene& operator=(const TexturedObjScene&) = delete;
        TexturedObjScene(TexturedObjScene&&) noexcept = default;
        TexturedObjScene& operator=(TexturedObjScene&&) noexcept = default;

        bool load(const std::filesystem::path& path);
        void draw(gfx::research::Shader& shader) const;
        void clear();

        [[nodiscard]] const glm::vec3& bounds_min() const { return bounds_min_; }
        [[nodiscard]] const glm::vec3& bounds_max() const { return bounds_max_; }
        [[nodiscard]] std::size_t submesh_count() const { return submeshes_.size(); }
        [[nodiscard]] std::size_t texture_count() const { return textures_.size(); }

    private:
        struct Material {
            glm::vec3 base_color{1.0f};
            float roughness = 0.55f;
            int texture_index = -1;
        };

        struct Submesh {
            gfx::research::Mesh mesh;
            int material_index = -1;
        };

        std::vector<Material> materials_;
        std::vector<Submesh> submeshes_;
        std::vector<gfx::research::Texture2D> textures_;
        glm::vec3 bounds_min_{};
        glm::vec3 bounds_max_{};
    };
}
