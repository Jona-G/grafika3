//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd*M_PI), kd(_kd), ks(_ks), shininess(_shininess) {}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) : start(_start), dir(normalize(_dir)) {}
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material)
		:center(_center), radius(_radius) { material = _material; }

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};
struct Triangle : public Intersectable {
	vec3 r1, r2, r3, n;

	Triangle(const vec3& _r1, const vec3& _r2, const vec3& _r3, Material* _material)
		:r1(_r1), r2(_r2), r3(_r3), n(cross(r2-r1, r3-r1)) { material = _material; }

	Hit intersect(const Ray& ray) {
		Hit hit;
		float t = dot(r1 - ray.start, n) / dot(ray.dir, n);
		if (t <= 0) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = n;
		hit.material = material;
		if ((dot(cross(r2-r1, hit.position-r1), n) > 0) &&
		    (dot(cross(r3-r2, hit.position-r2), n) > 0) &&
		    (dot(cross(r1-r3, hit.position-r3), n) > 0)) return hit;
		Hit outsideHit;
		return outsideHit;
	}
};

struct Cone : public Intersectable {
	vec3 p, n;
	float h, alpha;

	Cone(const vec3& _p, float _h, float _alpha, Material* _material)
		:p(_p), h(_h), alpha(_alpha) { material = _material; }

	Hit intersect(const Ray& ray) {
		Hit hit;
	 //_____ FIX ___________________________________________
		n = length(ray.start + ray.dir - p) * cosf(alpha);//
	 //----- FIX -------------------------------------------
		float a = dot(ray.dir, n) - dot(ray.dir, ray.dir) * pow(cosf(alpha), 2);
		float b = 2 * dot(ray.dir, dot(ray.start - p, dot(n, n) - pow(cosf(alpha), 2)));
		float c = dot(ray.start - p, ray.start - p) * (dot(n, n) - pow(cosf(alpha), 2));
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float t1 = (-b + sqrtf(discr)) / 2.0f / a;
		float t2 = (-b - sqrtf(discr)) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = length(hit.position - p) * cosf(alpha);
		hit.material = material;
		if (dot(hit.position - p, hit.normal) >= 0 &&
			dot(hit.position - p, hit.normal) <= h) return hit;
		Hit outsideHit;
		return outsideHit;
	}
};

struct Cube {
	Material* material;
	std::vector<vec3> vtx;
	vec3 center = vec3(0.0f, 0.0f, 0.0f);
	float length = 0.2f, dist = sqrt(3) * length / 2;

public:
	Cube(Material* _material) :material(_material) { build(); }

	Cube(vec3 _center, float _length, Material* _material)
		:center(_center), length(_length), material(_material) { build(); }

	void build() {
		vtx.clear();

		vtx.push_back(vec3(center.x - dist, center.y - dist, center.z - dist));
		vtx.push_back(vec3(center.x + dist, center.y - dist, center.z - dist));
		vtx.push_back(vec3(center.x - dist, center.y - dist, center.z + dist));
		vtx.push_back(vec3(center.x + dist, center.y - dist, center.z + dist));
		vtx.push_back(vec3(center.x - dist, center.y + dist, center.z - dist));
		vtx.push_back(vec3(center.x + dist, center.y + dist, center.z - dist));
		vtx.push_back(vec3(center.x - dist, center.y + dist, center.z + dist));
		vtx.push_back(vec3(center.x + dist, center.y + dist, center.z + dist));
	}

	std::vector<Intersectable*> create(std::vector<Intersectable*> objects) {
		objects.push_back(new Triangle(vtx[0], vtx[1], vtx[2], material));
		objects.push_back(new Triangle(vtx[0], vtx[1], vtx[5], material));
		objects.push_back(new Triangle(vtx[0], vtx[2], vtx[6], material));
		objects.push_back(new Triangle(vtx[0], vtx[4], vtx[5], material));
		objects.push_back(new Triangle(vtx[0], vtx[4], vtx[6], material));
		objects.push_back(new Triangle(vtx[1], vtx[2], vtx[3], material));
		objects.push_back(new Triangle(vtx[1], vtx[3], vtx[7], material));
		objects.push_back(new Triangle(vtx[1], vtx[5], vtx[7], material));
		objects.push_back(new Triangle(vtx[2], vtx[3], vtx[6], material));
		objects.push_back(new Triangle(vtx[3], vtx[6], vtx[7], material));
		objects.push_back(new Triangle(vtx[4], vtx[5], vtx[6], material));
		objects.push_back(new Triangle(vtx[5], vtx[6], vtx[7], material));
		return objects;
	}

	void moveTo(vec3 _center) {
		vec3 oldCenter = center;
		center = _center;
		vec3 newCenter = oldCenter - center;

		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x - newCenter.x, oldv.y - newCenter.y, oldv.z - newCenter.z);
		}
	}

	void resize(float newLength) {
		float olddist = sqrt(3) * length / 2;

		length = newLength;
		dist = sqrt(3) * length / 2;

		float newdist = dist - olddist;

		printf("olddist: %f\nlength: %f\ndist: %f\nnewdist: %f\n", olddist, length, dist, newdist);

		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x + newdist, oldv.y + newdist, oldv.z + newdist);
		}
	}

	void print() {
		for (int i = 0; i < 8; i++)
			printf("%f %f %f\n", vtx[i].x, vtx[i].y, vtx[i].z);
		printf("\n");
	}

	void rotateX(float angle) {
		angle = angle * (M_PI / 180.0f);
		vec3 currentCenter = center;
		moveTo(vec3(0.0f, 0.0f, 0.0f));
		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x + currentCenter.x,
					 oldv.y * cosf(angle) - oldv.z * sinf(angle) + currentCenter.y,
					 oldv.z * cosf(angle) + oldv.y * sinf(angle) + currentCenter.z);
		}
	}

	void rotateY(float angle) {
		angle = angle * (M_PI / 180.0f);
		vec3 currentCenter = center;
		moveTo(vec3(0.0f, 0.0f, 0.0f));
		for (auto& v : vtx) {
			vec3 oldv = v;
			v = vec3(oldv.x * cosf(angle) - oldv.z * sinf(angle),
					 oldv.y,
					 oldv.z * cosf(angle) + oldv.x * sinf(angle));
		}
		moveTo(currentCenter);
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) :  direction(normalize(_direction)), Le(_Le) {}
};

float rnd() { return (float)rand() / RAND_MAX; }
const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 0, 2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		//vec3 kd(0.5f, 0.5f, 0.5f), ks(2, 2, 2);
		//Material* material = new Material(kd, ks, 50);
		//vec3 t1, t2, t3;

		for (int i = 0; i < 60; i++) {
			vec3 kd(rnd() / 4 + 0.2f, rnd() / 4 + 0.2f, rnd() / 4 + 0.2f), ks(2, 2, 2);
			Material* material = new Material(kd, ks, 1);
			Cube cube = Cube(vec3(rnd() - 1.0f, 2 * rnd() - 1.0f, 2 * rnd() - 2.0f), 0.075f, material);
			cube.rotateX(rnd() * 60.0f);
			cube.rotateY(rnd() * 60.0f);
			objects = cube.create(objects);
		}
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	Hit secondIntersect(Ray ray) {
		Hit firstHit, secondHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (firstHit.t < 0 || hit.t < firstHit.t))  firstHit = hit;
			if (hit.t > 0 && (secondHit.t < 0 || hit.t > secondHit.t)) secondHit = hit;
		}
		if (dot(ray.dir, firstHit.normal) > 0) firstHit.normal = firstHit.normal * (-1);
		if (dot(ray.dir, secondHit.normal) > 0) secondHit.normal = secondHit.normal * (-1);
		return secondHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		//Hit hit = secondIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	srand(glutGet(GLUT_ELAPSED_TIME));
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}