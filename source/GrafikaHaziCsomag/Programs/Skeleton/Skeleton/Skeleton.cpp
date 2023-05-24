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

//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
	float f; // function value
	T d;  // derivatives
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 20;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extrinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	} 
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
			        0,                      1 / tan(fov / 2), 0,                      0,
			        0,                      0,                -(fp + bp) / (bp - fp), -1,
			        0,                      0,                -2 * fp*bp / (bp - fp),  0);
	}
};

//---------------------------
struct Material {
//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

//---------------------------
struct Light {
//---------------------------
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------
struct RenderState {
//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	Texture *          texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class GouraudShader : public Shader {
//---------------------------
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class Geometry {
//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
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
class ParamSurface : public Geometry {
//---------------------------
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
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
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};


//---------------------------
class Torus : public ParamSurface {
//---------------------------
public:
	Torus() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		const float R = 1, r = 0.5f;
		U = U * 2.0f * M_PI, V = V * 2.0f * M_PI;
		Dnum2 D = Cos(U) * r + R;
		X = D * Cos(V); Y = D * Sin(V); Z = Sin(U) * r;
	}
};

//---------------------------
struct Object {
//---------------------------
	Shader *   shader;
	Material * material;
	Geometry * geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader * _shader, Material * _material, Geometry * _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}
};

//---------------------------
class Scene {
//---------------------------
	std::vector<Object *> objects;
	Camera camera;
	std::vector<Light> lights;
public:
	void Build() {
		// Shaders
		Shader * gouraudShader = new GouraudShader();

		// Materials
		Material * material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		// Geometries
		Geometry * torus = new Torus();

		// Create objects by setting up their vertex data on the GPU

		Object * torusObject1 = new Object(gouraudShader, material0, torus);
		torusObject1->translation = vec3(0, 0, 0);
		torusObject1->scale = vec3(0.7f, 0.7f, 0.7f);
		torusObject1->rotationAxis = vec3(1, 0, 0);
		objects.push_back(torusObject1);

		// Lights
		lights.resize(3);
		lights[0].wLightPos = vec4(5, 5, 4, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(3, 0, 0);

		lights[1].wLightPos = vec4(5, 10, 20, 0);	// ideal point -> directional light source
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		lights[2].wLightPos = vec4(-5, 5, 5, 0);	// ideal point -> directional light source
		lights[2].La = vec3(0.1f, 0.1f, 0.1f);
		lights[2].Le = vec3(0, 0, 3);
	}

	void Render(Camera camera) {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}
};

Scene scene;
Camera jumperCamera, spinCamera;
bool animateStart = false;
float animateStartTime = 0.0f;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
	spinCamera.wEye = vec3(0, 0, 4);
	spinCamera.wLookat = vec3(0, 0, -1);
	spinCamera.wVup = vec3(0, 1, 0);

	jumperCamera.wEye = vec3(0, 3, 3);
	jumperCamera.wLookat = vec3(0, -1, -1);
	jumperCamera.wVup = vec3(0, 1, -1);
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, 0, windowWidth / 2, windowHeight);
	scene.Render(jumperCamera);

	glViewport(windowWidth / 2, 0, windowWidth / 2, windowHeight);
	scene.Render(spinCamera);

	glutSwapBuffers();
}

void onIdle() {
	static float dt = 0;
	dt = (glutGet(GLUT_ELAPSED_TIME) - animateStartTime) / 1000.0f;
	if (animateStart) {
		spinCamera.wEye = vec3(-sinf(dt), 0, cosf(dt)) * 4;
		spinCamera.wLookat = vec3(-sinf(dt), 0, cosf(dt));
	}
	glutPostRedisplay();
}

void onKeyboard(unsigned char key, int pX, int pY) { 
	animateStart = true; animateStartTime = glutGet(GLUT_ELAPSED_TIME);
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) { }


