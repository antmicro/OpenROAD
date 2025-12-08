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
 wire n_5_o0;
 wire n_6_o0;
 wire n_7_o0;
 wire n_8_o0;

 DFFHQNx1_ASAP7_75t_R DFF_A (.CLK(clk),
    .D(a),
    .QN(a_reg));
 DFFHQNx1_ASAP7_75t_R DFF_B (.CLK(clk),
    .D(b_reg1),
    .QN(b_reg2));
 INVx1_ASAP7_75t_R n_8 (.A(b),
    .Y(n_8_o0));
 INVx1_ASAP7_75t_R n_7 (.A(b_reg2),
    .Y(n_7_o0));
 BUFx2_ASAP7_75t_R n_6 (.A(a_reg),
    .Y(n_6_o0));
 INVx1_ASAP7_75t_R n_5 (.A(a_reg),
    .Y(n_5_o0));
endmodule
