import * as THREE from 'three';

const A = new THREE.Matrix4().makeRotationX(-Math.PI / 2);
const AI = A.clone().invert();
const asset = (path) => new URL(`../../embodiments/assets/${path}`, import.meta.url).href;
const suffix = (name) => name.split('/').at(-1);
const simVector = (v) => new THREE.Vector3(v[0], v[1], v[2]);
const converted = (matrix) => A.clone().multiply(matrix).multiply(AI);

function armFrame(head, tail, lateral) {
  const x = tail.clone().sub(head).normalize();
  const y = lateral.clone().addScaledVector(x, -lateral.dot(x));
  if (y.lengthSq() < 1e-12) y.crossVectors(x, Math.abs(x.x) < .8 ? new THREE.Vector3(1,0,0) : new THREE.Vector3(0,1,0));
  y.normalize();
  return new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x,y)).setPosition(head);
}

export class HumanPreview {
  static async load(scene, loader, replay, objects) {
    const wristIndex = replay.actors.findIndex((name) => suffix(name) === 'bone_00_wrist_root');
    if (wristIndex < 0) return null;
    const prefix = replay.actors[wristIndex].slice(0, -'bone_00_wrist_root'.length);
    const [figure, rig, placementResponse, bindResponse] = await Promise.all([
      loader.loadAsync(asset('artec_figure/standing.glb')),
      loader.loadAsync(asset('artec_hand/preview.glb')),
      fetch(asset('artec_figure/placement.json')),
      fetch(asset('artec_hand/preview_bind.json')),
    ]);
    if (!placementResponse.ok || !bindResponse.ok) throw new Error('Could not load human preview placement.');
    const placement = await placementResponse.json();
    const bind = await bindResponse.json();
    const instance = new HumanPreview();
    instance.indices = new Map(replay.actors.map((name,i) => [name.startsWith(prefix) ? suffix(name) : name,i]));
    const anchor = simVector(placement.standing.right_shoulder);
    const wrist = simVector(replay.frames[0][wristIndex]);
    const upperIndex = instance.indices.get('human_upper_arm');
    instance.inferred = upperIndex === undefined;
    instance.shoulder = instance.inferred ? new THREE.Vector3(wrist.x+.22,wrist.y,anchor.z) : simVector(replay.frames[0][upperIndex]);
    const rotation = new THREE.Matrix4().makeRotationZ(Math.PI/2);
    const origin = instance.shoulder.clone().sub(anchor.applyMatrix4(rotation));
    const figureWorld = rotation.setPosition(origin);
    figure.scene.applyMatrix4(converted(figureWorld));
    scene.add(figure.scene, rig.scene);
    for (const root of [figure.scene, rig.scene]) root.traverse((o) => {
      if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; o.frustumCulled = false; }
    });
    scene.updateMatrixWorld(true);
    instance.bones = [];
    rig.scene.traverse((bone) => {
      if (!bone.isBone || !bind[bone.name]) return;
      const actorBind = converted(new THREE.Matrix4().set(...bind[bone.name].flat()));
      instance.bones.push({bone, offset:actorBind.clone().invert().multiply(bone.matrixWorld),
        length: new THREE.Vector3().setFromMatrixColumn(actorBind,0).length()});
    });
    instance.bind = bind;
    for (const [name, object] of objects) {
      const link = suffix(name);
      if (name.startsWith(prefix) && (link.startsWith('bone_') || ['human_upper_arm','human_forearm','human_shoulder_anchor'].includes(link))) object.visible = false;
    }
    return instance;
  }

  update(frameA, frameB, mix) {
    const worlds = new Map();
    for (const [name,index] of this.indices) {
      const a=frameA[index], b=frameB[index];
      const position=simVector(a).lerp(simVector(b),mix);
      const q=new THREE.Quaternion(...a.slice(3,7)).slerp(new THREE.Quaternion(...b.slice(3,7)),mix);
      worlds.set(name,new THREE.Matrix4().compose(position,q,new THREE.Vector3(1,1,1)));
    }
    if (this.inferred) {
      const wristWorld=worlds.get('bone_00_wrist_root');
      const wrist=new THREE.Vector3().setFromMatrixPosition(wristWorld);
      const delta=wrist.clone().sub(this.shoulder), distance=Math.max(delta.length(),1e-8), direction=delta.divideScalar(distance);
      const pole=new THREE.Vector3(0,1,-.35); pole.addScaledVector(direction,-pole.dot(direction));
      if (pole.lengthSq()<1e-12) pole.crossVectors(direction,new THREE.Vector3(1,0,0));
      pole.normalize();
      const extension=Math.max(1,distance/.609), upper=.32*extension, fore=.29*extension;
      const along=THREE.MathUtils.clamp((upper*upper-fore*fore+distance*distance)/(2*distance),-upper,upper);
      const elbow=this.shoulder.clone().addScaledVector(direction,along).addScaledVector(pole,Math.sqrt(Math.max(0,upper*upper-along*along)));
      const lateral=new THREE.Vector3().setFromMatrixColumn(wristWorld,0);
      for (const [name,head,tail,next] of [
        ['human_upper_arm',this.shoulder,elbow,'human_forearm'],
        ['human_forearm',elbow,wrist,'bone_00_wrist_root'],
      ]) {
        const matrix=armFrame(head,tail,lateral);
        const bindHead=new THREE.Vector3(...this.bind[name].slice(0,3).map((r)=>r[3]));
        const bindTail=new THREE.Vector3(...this.bind[next].slice(0,3).map((r)=>r[3]));
        matrix.scale(new THREE.Vector3(head.distanceTo(tail)/bindHead.distanceTo(bindTail),1,1));
        worlds.set(name,matrix);
      }
    }
    for (const {bone,offset} of this.bones) {
      const world=worlds.get(bone.name);
      if (!world) continue;
      const local=bone.parent.matrixWorld.clone().invert().multiply(converted(world)).multiply(offset);
      local.decompose(bone.position,bone.quaternion,bone.scale);
      bone.updateMatrixWorld(true);
    }
  }
}
