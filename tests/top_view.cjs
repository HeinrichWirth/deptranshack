const assert = require('node:assert/strict');
const math = require('../web/viewer/predicted_train_math.js');

// Physical sides must agree in both views, independent of world heading.
for (const angle of [0, 0.6, -1.2, Math.PI]) {
  const rotate = ([x,y,z]) => [x*Math.cos(angle)-y*Math.sin(angle), x*Math.sin(angle)+y*Math.cos(angle), z];
  const curves = {
    left: [[-.8,0,0],[-.8,-5,0],[-.8,-10,0]].map(rotate),
    right: [[.8,0,0],[.8,-5,0],[.8,-10,0]].map(rotate)
  };
  const pose = math.sample(math.build(curves), 0);
  for (const side of [-2, 2]) {
    const point = rotate([side,-4,0]);
    const top = math.topView(point, pose);
    const slice = math.project(point, pose);
    assert.ok(Math.abs(top[0]-side)<1e-10);
    assert.ok(Math.abs(top[1]-4)<1e-10);
    assert.equal(Math.sign(top[0]), Math.sign(slice[0]));
  }
}
console.log('Top view: physical sides and forward direction agree with slice.');
