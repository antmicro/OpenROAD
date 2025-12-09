module sample (a,
    a_inv1,
    a_inv2,
    b,
    b_inv1,
    clk);
 input a;
 output a_inv1;
 output a_inv2;
 input b;
 output b_inv1;
 input clk;

 wire a_reg;
 wire b_reg1;
 wire b_reg2;

 DFFHQNx1_ASAP7_75t_R DFF_A (.CLK(clk),
    .D(a),
    .QN(a_reg));
 DFFHQNx1_ASAP7_75t_R DFF_B (.CLK(clk),
    .D(b_reg1),
    .QN(b_reg2));
 INVx1_ASAP7_75t_R n_8 (.A(b),
    .Y(b_reg1));
 INVx1_ASAP7_75t_R n_7 (.A(b_reg2),
    .Y(b_inv1));
 BUFx2_ASAP7_75t_R n_6 (.A(a_reg),
    .Y(a_inv2));
 INVx1_ASAP7_75t_R n_5 (.A(a_reg),
    .Y(a_inv1));
endmodule
