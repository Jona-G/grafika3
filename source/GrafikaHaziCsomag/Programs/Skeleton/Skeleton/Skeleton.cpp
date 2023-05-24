//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
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
// Nev    : Pribék Jonatán
// Neptun : P1PGS0
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

inline float PMRND() { return 2.0f * rand() / RAND_MAX - 1.0f; }
inline float Rand(float mean, float var) { return mean + PMRND() * var; }

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess, emission;

	Material() { ka = vec3(1, 1, 1), kd = ks = vec3(0, 0, 0); shininess = 1; emission = 0; }
};

//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material* material, const std::string& name) {
		if (material != NULL) {
			setUniform(material->kd, name + ".kd");
			setUniform(material->ks, name + ".ks");
			setUniform(material->ka, name + ".ka");
			setUniform(material->shininess, name + ".shininess");
			setUniform(material->emission, name + ".emission");
		}
		else {
			setUniform(vec3(0, 0, 0), name + ".kd");
			setUniform(vec3(0, 0, 0), name + ".ks");
			setUniform(vec3(1, 1, 1), name + ".ka");
			setUniform(0, name + ".shininess");
			setUniform(0, name + ".emission");
		}
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class VolumetricShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4 MVP;								 // MVP
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 2) in vec2  vtxUV;
		layout(location = 3) in vec4  vtxColor;      	 // modulation color
		out vec4 modulation;
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		    modulation = vtxColor;
		    texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D textureMap;
		in vec4 modulation;
		in vec2 texcoord;	
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = texture(textureMap, texcoord) * modulation;
		}
	)";
public:
	VolumetricShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(*state.texture, "textureMap");
	}
};

//---------------------------
class SurfaceShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye
		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;
		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess, emission;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;
		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces 
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = texColor * material.emission;
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	SurfaceShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
	vec4 color;
};

//---------------------------
struct Geometry {
	//---------------------------
	unsigned int vao, vbo;
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
struct TriangleMesh : public Geometry {
	//---------------------------
	std::vector<VertexData> mesh;

	void create() {
		glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(VertexData), &mesh[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		glEnableVertexAttribArray(3);  // attribute array 3 = COLOR
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, color));
	}
	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, mesh.size());
	}
};

//---------------------------
struct Quad : public TriangleMesh {
	//---------------------------
	Quad() {
		VertexData vd1, vd2, vd3, vd4;
		vd1.normal = vd2.normal = vd3.normal = vd4.normal = vec3(0, 0, 1);
		vd1.color = vd2.color = vd3.color = vd4.color = vec4(1, 1, 1, 1);
		vd1.texcoord = vec2(0, 0); vd2.texcoord = vec2(1, 0);
		vd3.texcoord = vec2(1, 1); vd4.texcoord = vec2(0, 1);
		vd1.position = vec3(-1, -1, 0); vd2.position = vec3(1, -1, 0);
		vd3.position = vec3(1, 1, 0);   vd4.position = vec3(-1, 1, 0);
		mesh.push_back(vd1); mesh.push_back(vd2); mesh.push_back(vd3);
		mesh.push_back(vd1); mesh.push_back(vd3); mesh.push_back(vd4);
		create();
	}
};

//---------------------------
struct BoxSurface : public TriangleMesh {
	//---------------------------
	void PushQuad(VertexData vd[4]) {
		mesh.push_back(vd[0]); mesh.push_back(vd[1]); mesh.push_back(vd[2]);
		mesh.push_back(vd[0]); mesh.push_back(vd[2]); mesh.push_back(vd[3]);
	}
	BoxSurface(float s) {
		vec3 p1(-s, -s, -s), p2(s, -s, -s), p3(s, s, -s), p4(-s, s, -s),
			p5(-s, -s, s), p6(s, -s, s), p7(s, s, s), p8(-s, s, s);
		VertexData vd[4];
		vd[0].texcoord = vec2(0, 0); vd[1].texcoord = vec2(1, 0); vd[2].texcoord = vec2(1, 1); vd[2].texcoord = vec2(0, 1);
		vd[0].color = vd[1].color = vd[2].color = vd[3].color = vec4(0, 0, 1, 1);
		vd[0].normal = vd[1].normal = vd[2].normal = vd[3].normal = vec3(0, 0, 1);
		vd[0].position = p1; vd[1].position = p2; vd[2].position = p3; vd[3].position = p4; PushQuad(vd);
		vd[0].normal = vd[1].normal = vd[2].normal = vd[3].normal = vec3(0, 0, -1);
		vd[0].position = p5; vd[1].position = p6; vd[2].position = p7; vd[3].position = p8; PushQuad(vd);
		vd[0].color = vd[1].color = vd[2].color = vd[3].color = vec4(0, 1, 0, 1);
		vd[0].normal = vd[1].normal = vd[2].normal = vd[3].normal = vec3(0, 1, 0);
		vd[0].position = p1; vd[1].position = p2; vd[2].position = p6; vd[3].position = p5; PushQuad(vd);
		vd[0].normal = vd[1].normal = vd[2].normal = vd[3].normal = vec3(0, -1, 0);
		vd[0].position = p3; vd[1].position = p4; vd[2].position = p8; vd[3].position = p7; PushQuad(vd);
		vd[0].color = vd[1].color = vd[2].color = vd[3].color = vec4(1, 0, 0, 1);
		vd[0].normal = vd[1].normal = vd[2].normal = vd[3].normal = vec3(1, 0, 0);
		vd[0].position = p1; vd[1].position = p4; vd[2].position = p8; vd[3].position = p5; PushQuad(vd);
		vd[0].normal = vd[1].normal = vd[2].normal = vd[3].normal = vec3(-1, 0, 0);
		vd[0].position = p2; vd[1].position = p3; vd[2].position = p7; vd[3].position = p6; PushQuad(vd);
		create();
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }
	virtual VertexData GenVertexData(float u, float v) = 0;

	void create(int N = 40, int M = 40) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

//---------------------------
class Sphere : public ParamSurface {
	//---------------------------
public:
	Sphere() { create(); }
	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position = vd.normal = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			cosf(v * (float)M_PI));
		vd.texcoord = vec2(u, v);
		return vd;
	}
};

//---------------------------
struct Ray {
	//---------------------------
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = _dir; }
};

//---------------------------
struct GameObject {
	//---------------------------
	virtual Shader* getShader() { return NULL; }
	virtual Material* getMaterial() { return NULL; }
	virtual Texture* getTexture() { return NULL; }
	virtual Geometry* getGeometry() { return NULL; }

	vec3   position, velocity, acceleration;
	bool   alive;
	float  boundingRadius;

	GameObject() : position(0, 0, 0), velocity(0, 0, 0), acceleration(0, 0, 0) {
		alive = true;
		boundingRadius = 0;
	}
	virtual void Control(float tstart, float tend) { }      // control
	virtual void Animate(float tstart, float tend) {        // state change
		float dt = tend - tstart;
		position = position + velocity * dt;          // Euler integration
		velocity = velocity + acceleration * dt;
	}
	virtual void SetModelingTransformation(RenderState& state) {
		state.M = TranslateMatrix(position);
		state.Minv = TranslateMatrix(-position);
	}
	virtual void Draw(RenderState state) {
		if (!getShader()) return;
		SetModelingTransformation(state);
		state.MVP = state.M * state.V * state.P;
		state.material = getMaterial();
		state.texture = getTexture();
		getShader()->Bind(state);
		getGeometry()->Draw();
	}
	virtual void Interact(GameObject* obj) { }
	virtual bool Collide(GameObject* obj, float& hit_time, vec3& hit_point) {
		float radius = boundingRadius + obj->boundingRadius;
		Ray ray(position, velocity - obj->velocity);
		float this_hit_time = obj->Intersect(ray, radius);
		if (this_hit_time > 0 && this_hit_time < hit_time) {
			hit_time = this_hit_time;
			vec3 hit_pos = position + velocity * hit_time;
			vec3 obj_hit_pos = obj->position + obj->velocity * hit_time;
			float a = boundingRadius / (boundingRadius + obj->boundingRadius);
			hit_point = hit_pos * (1 - a) + obj_hit_pos * a;
			return true;
		}
		else return false;
	}
	float Intersect(const Ray& ray, float radius) { // ray - sphere intersection
		if (radius == 0) return false;
		vec3 dist = ray.start - position;
		float a = dot(ray.dir, ray.dir), b = dot(dist, ray.dir) * 2.0f, c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return false;
		float sqrt_discr = (float)sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// larger
		float t2 = (-b - sqrt_discr) / 2.0f / a;	// smaller
		if (t2 > 0) return t2;
		if (t1 > 0) return t1;
		return -1;
	}
	virtual void Kill() { alive = false; }
};

//--------------------------------------------
struct Input {
	//--------------------------------------------
	bool    glutKeyTable[256]; // key table
	Input() { for (int i = 0; i < 256; i++) glutKeyTable[i] = false; }
	bool GetKeyStatus(int keyCode) { return glutKeyTable[keyCode]; }
};

extern Input input;

//--------------------------------------------
struct Avatar : public GameObject {
	//--------------------------------------------
	vec3  wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic

	Avatar(vec3& pos) {
		position = pos;
		wVup = vec3(0, 0, 1);
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.1f; bp = 1000.0f;
	}
	virtual void ProcessInput() = 0;

	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(-velocity);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(-position) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}
	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

//---------------------------
class Scene {
	//---------------------------
	int current = 0;
	std::vector<GameObject*> objects[2]; // ping-pong
	std::vector<Light>        lights;
public:
	Avatar* avatar;

	void Build();

	void Render() {
		RenderState state;
		state.wEye = avatar->position;
		state.V = avatar->V();
		state.P = avatar->P();
		state.lights = lights;
		// First pass: opaque objects
		for (auto* obj : objects[current]) if (dynamic_cast<SurfaceShader*>(obj->getShader())) obj->Draw(state);

		// Second pass: transparent objects
		glEnable(GL_BLEND);
		glDepthMask(GL_FALSE);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glEnable(GL_ALPHA_TEST);			// only non-zero alpha
		glAlphaFunc(GL_GREATER, 0);
		for (auto* obj : objects[current]) if (dynamic_cast<VolumetricShader*>(obj->getShader())) obj->Draw(state);
		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
	}

	void Simulate(float tstart, float tend) {
		avatar->ProcessInput();
		const float dt = 0.1f; // dt is ”infinitesimal”
		for (float t = tstart; t < tend; t += dt) {
			float Dt = fmin(dt, tend - t);
			for (auto* obj : objects[current]) obj->Control(tstart, tend); // control
			for (auto* obj : objects[current]) {
				if (obj->alive) Join(obj);
				else            delete obj; // bury dead
			}
			objects[current].clear();
			current = !current;
			for (auto* obj : objects[current]) obj->Animate(tstart, tend); // animate
		}
		Render();
	}

	void Interact(GameObject* object) {
		for (auto* obj : objects[current]) object->Interact(obj);
	}

	GameObject* Collide(GameObject* object, float& hit_time, vec3& hit_point) {
		GameObject* hit_object = NULL;
		for (auto* obj : objects[current]) {
			if (obj == object) continue;
			if (obj->Collide(object, hit_time, hit_point)) hit_object = obj;
		}
		return hit_object;
	}
	void Join(GameObject* obj) { objects[!current].push_back(obj); }
};

extern Scene* scene;

//--------------------------------------------
class BillBoard : public GameObject {
	//--------------------------------------------
protected:
	float size;
public:
	static Shader* shader;
	static Geometry* geometry;

	BillBoard(vec3& _position, float size0) {
		position = _position;
		size = size0;
	}
	Shader* getShader() { return shader; }
	Geometry* getGeometry() { return geometry; }

	void SetModelingTransformation(RenderState& state) {
		vec3 w = state.wEye - position;
		vec3 left = cross(vec3(0, 0, 1), w);
		vec3 up = cross(w, left);
		left = normalize(left) * size;
		up = normalize(up) * size;
		state.M = mat4(left.x, left.y, left.z, 0,
			up.x, up.y, up.z, 0,
			0, 0, 1, 0,
			0, 0, 0, 1) * TranslateMatrix(position);
	}
};

//--------------------------------------------
struct Particle {
	//--------------------------------------------
	vec3	position, velocity, acceleration;
	float   weight, dweight, size, dsize, time_to_live;
	vec4    color, dcolor;

	Particle() { weight = 1; }

	void Animate(float dt, vec3& force) {
		time_to_live -= dt;
		if (time_to_live <= 0) return;
		acceleration = force * (1 / weight);
		velocity = velocity + acceleration * dt;
		position = position + velocity * dt;
		weight += dweight * dt;
		size += dsize * dt;
		color += dcolor * dt;
		if (color.w <= 0) time_to_live = 0;
	}
};

//--------------------------------------------
class ParticleSystem : public GameObject {
	//--------------------------------------------
protected:
	unsigned int vao, vbo;
	std::vector<Particle*> particles;
	float				   age;
	vec3				   force;
public:
	static Shader* shader;
	static Texture* texture;

	ParticleSystem(vec3& pos0) {
		position = pos0; age = 0.0; alive = true;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		glEnableVertexAttribArray(3);  // attribute array 3 = COLOR
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, color));
	}
	Shader* getShader() { return shader; }
	Texture* getTexture() { return texture; }

	void  Control(float tstart, float tend) {
		alive = false;
		for (auto particle : particles) {
			if (particle->time_to_live > 0) {
				alive = true;
				return;
			}
		}
	}

	void  Animate(float tstart, float tend) {
		float dt = tend - tstart;
		for (auto particle : particles) particle->Animate(dt, force);
	}

	void AddParticle(Particle* particle) { particles.push_back(particle); }

	void Draw(RenderState state) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		std::vector<VertexData> mesh;
		VertexData vd1, vd2, vd3, vd4;
		vd1.normal = vd2.normal = vd3.normal = vd4.normal = vec3(0, 0, 1);
		vd1.texcoord = vec2(0, 0); vd2.texcoord = vec2(1, 0);
		vd3.texcoord = vec2(1, 1); vd4.texcoord = vec2(0, 1);

		for (auto particle : particles) {
			vec3 p = position + particle->position;
			vec3 w = state.wEye - p;
			vec3 left = cross(vec3(0, 0, 1), w);
			vec3 up = cross(w, left);
			left = normalize(left) * particle->size;
			up = normalize(up) * particle->size;
			vd1.color = vd2.color = vd3.color = vd4.color = particle->color;
			vd1.position = p - left - up;
			vd2.position = p + left - up;
			vd3.position = p + left + up;
			vd4.position = p - left + up;
			mesh.push_back(vd1); mesh.push_back(vd2); mesh.push_back(vd3);
			mesh.push_back(vd1); mesh.push_back(vd3); mesh.push_back(vd4);
		}
		glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(VertexData), &mesh[0], GL_DYNAMIC_DRAW);
		state.MVP = state.V * state.P;
		state.texture = texture;
		shader->Bind(state);
		glDrawArrays(GL_TRIANGLES, 0, mesh.size());
	}
	~ParticleSystem() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
		for (auto particle : particles) delete particle;
	}
};

Input input;
const float GAME_SPEED = 1000.0f;

//-----------------------------------------------------------------
void onDisplay(void) {
	//-----------------------------------------------------------------
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene->Render();
	glutSwapBuffers();
}

//-----------------------------------------------------------------
void onIdle(void) {
	//-----------------------------------------------------------------
	static float tend = 0;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / GAME_SPEED;
	scene->Simulate(tstart, tend);
	glutPostRedisplay();
}

//-----------------------------------------------------------------
void onKeyboard(unsigned char key, int x, int y) { // ascii chars
	//-----------------------------------------------------------------
	input.glutKeyTable[key] = true;
}

//-----------------------------------------------------------------
void onKeyboardUp(unsigned char key, int x, int y) { // ascii chars
	//-----------------------------------------------------------------
	input.glutKeyTable[key] = false;
}

//-----------------------------------------------------------------
void onSpecialKey(int key, int x, int y) { // not ascii chars
	//-----------------------------------------------------------------
	input.glutKeyTable[key] = true;
}

//-----------------------------------------------------------------
void onSpecialKeyUp(int key, int x, int y) { // not ascii chars
	//-----------------------------------------------------------------
	input.glutKeyTable[key] = false;
}

Scene* scene;

Geometry* BillBoard::geometry = NULL;
Shader* BillBoard::shader = NULL;
Shader* ParticleSystem::shader = NULL;
Texture* ParticleSystem::texture = NULL;

// Initialization, create an OpenGL context
//-----------------------------------------------------------------
void onInitialization() {
	//-----------------------------------------------------------------
	BillBoard::shader = ParticleSystem::shader = new VolumetricShader();
	BillBoard::geometry = new Quad();
	ParticleSystem::texture = new Texture("explosion.bmp", true);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene = new Scene;
	scene->Build();
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Entry point of the application
int main(int argc, char* argv[]) {
	// Initialize GLUT, Glew and OpenGL 
	glutInit(&argc, argv);

	// OpenGL major and minor versions
	int majorVersion = 3, minorVersion = 3;
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif
	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Initialize this program and create shaders
	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutSpecialFunc(onSpecialKey);
	glutSpecialUpFunc(onSpecialKeyUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	return 1;
}
