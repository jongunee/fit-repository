// /static/js/fitting_room_viewer.js

function load3DModel(objUrl) {
  const container = document.getElementById('3d-container');
  const existingCanvas = container.querySelector('canvas');
  if (existingCanvas) {
      container.removeChild(existingCanvas);
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);

  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setClearColor(0xffffff);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 1;
  controls.maxDistance = 500;
  controls.maxPolarAngle = Math.PI / 2;

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(0, 1, 1);
  scene.add(directionalLight);

  const loader = new THREE.OBJLoader();
  loader.load(objUrl, function (object) {
      object.traverse(function (child) {
          if (child instanceof THREE.Mesh) {
              child.material = new THREE.MeshPhongMaterial({ color: 0xdcdcdc, specular: 0x111111, shininess: 100 });
          }
      });

      const box = new THREE.Box3().setFromObject(object);
      const size = box.getSize(new THREE.Vector3()).length();
      const center = box.getCenter(new THREE.Vector3());

      object.position.x += (object.position.x - center.x);
      object.position.y += (object.position.y - center.y);
      object.position.z += (object.position.z - center.z);

      object.scale.set(3, 3, 3);
      scene.add(object);

      camera.position.set(0, 0, size * 2);

      animate();
  }, undefined, function (error) {
      console.error("An error happened during loading:", error);
  });

  function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
  }
}
