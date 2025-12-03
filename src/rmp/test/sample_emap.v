module sample (a,
    a_inv1,
    a_inv2,
    b,
    b_inv1,
    clk,
    pi_0,
    pi_1,
    pi_2,
    po_0,
    po_1,
    po_2,
    po_3);
 input a;
 output a_inv1;
 output a_inv2;
 input b;
 output b_inv1;
 input clk;
 input pi_0;
 input pi_1;
 input pi_2;
 output po_0;
 output po_1;
 output po_2;
 output po_3;

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
 INVx1_ASAP7_75t_R n_8 (.A(pi_1),
    .Y(n_8_o0));
 INVx1_ASAP7_75t_R n_7 (.A(pi_2),
    .Y(n_7_o0));
 BUFx2_ASAP7_75t_R n_6 (.A(pi_0),
    .Y(n_6_o0));
 INVx1_ASAP7_75t_R n_5 (.A(pi_0),
    .Y(n_5_o0));
 assign po_0 = n_5_o0;
 assign po_1 = n_6_o0;
 assign po_2 = n_7_o0;
 assign po_3 = n_8_o0;
endmodule
