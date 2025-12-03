/*
a -> DFF_A -> INV_A_0 -> a_inv1
                      -> INV_A_1 -> a_inv2

b -> INV_B_0 -> DFF_B -> INV_B_1 -> a_inv1
*/

module sample (
  clk,
  a,
  b,
  a_inv1,
  a_inv2,
  b_inv1
);
  input clk;
  input a;
  input b;
  output a_inv1;
  output a_inv2;
  output b_inv1;

  wire a_reg;
  wire b_reg1;
  wire b_reg2;

  DFFHQNx1_ASAP7_75t_R DFF_A (
    .CLK(clk),
    .D(a),
    .QN(a_reg)
  );

  INVx1_ASAP7_75t_R INV_A_0 (
    .A(a_reg),
    .Y(a_inv1)
  );

  INVx1_ASAP7_75t_R INV_A_1 (
    .A(a_inv1),
    .Y(a_inv2)
  );

  INVx1_ASAP7_75t_R INV_B_0 (
    .A(b),
    .Y(b_reg1)
  );

  DFFHQNx1_ASAP7_75t_R DFF_B (
    .CLK(clk),
    .D(b_reg1),
    .QN(b_reg2)
  );

  INVx1_ASAP7_75t_R INV_B_1 (
    .A(b_reg2),
    .Y(b_inv1)
  );

endmodule
