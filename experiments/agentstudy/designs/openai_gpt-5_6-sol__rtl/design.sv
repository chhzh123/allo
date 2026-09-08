module dut_norm_pe (
  input wire ap_clk, input wire ap_rst_n,
  input wire signed [7:0] a_in, input wire a_valid_in,
  input wire a_first_in, input wire a_last_in, input wire a_bank_in,
  output logic signed [7:0] a_out, output logic a_valid_out,
  output logic a_first_out, output logic a_last_out, output logic a_bank_out,
  input wire signed [7:0] b_in, input wire b_valid_in,
  output logic signed [7:0] b_out, output logic b_valid_out,
  input wire signed [31:0] result0_from_east, result1_from_east,
  input wire shift_result0, shift_result1,
  output wire signed [31:0] result0_to_west, result1_to_west,
  output wire final_event
);
  logic signed [15:0] product_pipe;
  logic product_valid, product_first, product_last, product_bank;
  logic signed [31:0] accumulator;
  logic signed [31:0] result0, result1;
  wire signed [31:0] product_extended = {{16{product_pipe[15]}},product_pipe};
  wire signed [31:0] accumulated_value = accumulator + product_extended;
  assign result0_to_west=result0;
  assign result1_to_west=result1;
  assign final_event=product_valid && product_last;

  always_ff @(posedge ap_clk) begin
    if (!ap_rst_n) begin
      a_out<='0; a_valid_out<=0; a_first_out<=0; a_last_out<=0; a_bank_out<=0;
      b_out<='0; b_valid_out<=0;
      product_pipe<='0; product_valid<=0; product_first<=0; product_last<=0; product_bank<=0;
      accumulator<='0; result0<='0; result1<='0;
    end else begin
      // Operand links are registered nearest-neighbour links.
      a_valid_out<=a_valid_in;
      if (a_valid_in) begin
        a_out<=a_in; a_first_out<=a_first_in; a_last_out<=a_last_in; a_bank_out<=a_bank_in;
      end
      b_valid_out<=b_valid_in;
      if (b_valid_in) b_out<=b_in;

      // Exactly one registered signed multiplier in this PE.
      product_valid<=a_valid_in && b_valid_in;
      if (a_valid_in && b_valid_in) begin
        product_pipe <= $signed(a_in)*$signed(b_in);
        product_first<=a_first_in; product_last<=a_last_in; product_bank<=a_bank_in;
      end

      // This PE alone accumulates and holds its output-stationary partial sum.
      if (product_valid) begin
        if (product_first) accumulator<=product_extended;
        else accumulator<=accumulated_value;
      end

      // Alternating result banks permit accumulation of the next product while
      // this product takes the registered westward row chain to its port.
      if (product_valid && product_last && !product_bank) result0<=accumulated_value;
      else if (shift_result0) result0<=result0_from_east;
      if (product_valid && product_last && product_bank) result1<=accumulated_value;
      else if (shift_result1) result1<=result1_from_east;
    end
  end
endmodule

module dut_norm (
  input wire ap_clk, input wire ap_rst_n,
  input wire [7:0] a_in_0_dout, input wire a_in_0_empty_n, output wire a_in_0_read,
  input wire [7:0] a_in_1_dout, input wire a_in_1_empty_n, output wire a_in_1_read,
  input wire [7:0] a_in_2_dout, input wire a_in_2_empty_n, output wire a_in_2_read,
  input wire [7:0] a_in_3_dout, input wire a_in_3_empty_n, output wire a_in_3_read,
  input wire [7:0] a_in_4_dout, input wire a_in_4_empty_n, output wire a_in_4_read,
  input wire [7:0] a_in_5_dout, input wire a_in_5_empty_n, output wire a_in_5_read,
  input wire [7:0] a_in_6_dout, input wire a_in_6_empty_n, output wire a_in_6_read,
  input wire [7:0] a_in_7_dout, input wire a_in_7_empty_n, output wire a_in_7_read,
  input wire [7:0] b_in_0_dout, input wire b_in_0_empty_n, output wire b_in_0_read,
  input wire [7:0] b_in_1_dout, input wire b_in_1_empty_n, output wire b_in_1_read,
  input wire [7:0] b_in_2_dout, input wire b_in_2_empty_n, output wire b_in_2_read,
  input wire [7:0] b_in_3_dout, input wire b_in_3_empty_n, output wire b_in_3_read,
  input wire [7:0] b_in_4_dout, input wire b_in_4_empty_n, output wire b_in_4_read,
  input wire [7:0] b_in_5_dout, input wire b_in_5_empty_n, output wire b_in_5_read,
  input wire [7:0] b_in_6_dout, input wire b_in_6_empty_n, output wire b_in_6_read,
  input wire [7:0] b_in_7_dout, input wire b_in_7_empty_n, output wire b_in_7_read,
  output wire [31:0] c_out_0_din, input wire c_out_0_full_n, output wire c_out_0_write,
  output wire [31:0] c_out_1_din, input wire c_out_1_full_n, output wire c_out_1_write,
  output wire [31:0] c_out_2_din, input wire c_out_2_full_n, output wire c_out_2_write,
  output wire [31:0] c_out_3_din, input wire c_out_3_full_n, output wire c_out_3_write,
  output wire [31:0] c_out_4_din, input wire c_out_4_full_n, output wire c_out_4_write,
  output wire [31:0] c_out_5_din, input wire c_out_5_full_n, output wire c_out_5_write,
  output wire [31:0] c_out_6_din, input wire c_out_6_full_n, output wire c_out_6_write,
  output wire [31:0] c_out_7_din, input wire c_out_7_full_n, output wire c_out_7_write
);
  wire [7:0] ad[0:7], bd[0:7]; wire ae[0:7],be[0:7],ar[0:7],br[0:7];
  assign ad[0]=a_in_0_dout; assign ae[0]=a_in_0_empty_n; assign a_in_0_read=ar[0];
  assign ad[1]=a_in_1_dout; assign ae[1]=a_in_1_empty_n; assign a_in_1_read=ar[1];
  assign ad[2]=a_in_2_dout; assign ae[2]=a_in_2_empty_n; assign a_in_2_read=ar[2];
  assign ad[3]=a_in_3_dout; assign ae[3]=a_in_3_empty_n; assign a_in_3_read=ar[3];
  assign ad[4]=a_in_4_dout; assign ae[4]=a_in_4_empty_n; assign a_in_4_read=ar[4];
  assign ad[5]=a_in_5_dout; assign ae[5]=a_in_5_empty_n; assign a_in_5_read=ar[5];
  assign ad[6]=a_in_6_dout; assign ae[6]=a_in_6_empty_n; assign a_in_6_read=ar[6];
  assign ad[7]=a_in_7_dout; assign ae[7]=a_in_7_empty_n; assign a_in_7_read=ar[7];
  assign bd[0]=b_in_0_dout; assign be[0]=b_in_0_empty_n; assign b_in_0_read=br[0];
  assign bd[1]=b_in_1_dout; assign be[1]=b_in_1_empty_n; assign b_in_1_read=br[1];
  assign bd[2]=b_in_2_dout; assign be[2]=b_in_2_empty_n; assign b_in_2_read=br[2];
  assign bd[3]=b_in_3_dout; assign be[3]=b_in_3_empty_n; assign b_in_3_read=br[3];
  assign bd[4]=b_in_4_dout; assign be[4]=b_in_4_empty_n; assign b_in_4_read=br[4];
  assign bd[5]=b_in_5_dout; assign be[5]=b_in_5_empty_n; assign b_in_5_read=br[5];
  assign bd[6]=b_in_6_dout; assign be[6]=b_in_6_empty_n; assign b_in_6_read=br[6];
  assign bd[7]=b_in_7_dout; assign be[7]=b_in_7_empty_n; assign b_in_7_read=br[7];

  logic [3:0] launch_time; logic [2:0] ak[0:7]; logic product_bank[0:7];
  genvar p;
  generate for(p=0;p<8;p=p+1) begin:ports
    assign ar[p]=(launch_time>=p); assign br[p]=(launch_time>=p);
  end endgenerate
  integer n;
  always_ff @(posedge ap_clk) begin
    if(!ap_rst_n) begin
      launch_time<=0;
      for(n=0;n<8;n=n+1) begin ak[n]<=0; product_bank[n]<=0; end
    end else begin
      if(launch_time!=15) launch_time<=launch_time+1'b1;
      for(n=0;n<8;n=n+1) if(ar[n]&&ae[n]) begin
        if(ak[n]==7) begin ak[n]<=0; product_bank[n]<=~product_bank[n]; end
        else ak[n]<=ak[n]+1'b1;
      end
    end
  end

  wire signed [7:0] ah[0:7][0:8], bv[0:8][0:7];
  wire ahv[0:7][0:8], aff[0:7][0:8], all[0:7][0:8], abb[0:7][0:8];
  wire bvv[0:8][0:7];
  generate for(p=0;p<8;p=p+1) begin:boundaries
    assign ah[p][0]=$signed(ad[p]); assign ahv[p][0]=ar[p]&&ae[p];
    assign aff[p][0]=(ak[p]==0); assign all[p][0]=(ak[p]==7); assign abb[p][0]=product_bank[p];
    assign bv[0][p]=$signed(bd[p]); assign bvv[0][p]=br[p]&&be[p];
  end endgenerate

  wire signed [31:0] res0[0:7][0:7],res1[0:7][0:7]; wire finished[0:7][0:7];
  logic active[0:7], outbank[0:7]; logic [2:0] outpos[0:7]; logic [1:0] pending[0:7];
  wire ready[0:7],take[0:7],sh0[0:7],sh1[0:7];
  assign ready[0]=c_out_0_full_n; assign c_out_0_write=active[0]; assign c_out_0_din=outbank[0]?res1[0][0]:res0[0][0];
  assign ready[1]=c_out_1_full_n; assign c_out_1_write=active[1]; assign c_out_1_din=outbank[1]?res1[1][0]:res0[1][0];
  assign ready[2]=c_out_2_full_n; assign c_out_2_write=active[2]; assign c_out_2_din=outbank[2]?res1[2][0]:res0[2][0];
  assign ready[3]=c_out_3_full_n; assign c_out_3_write=active[3]; assign c_out_3_din=outbank[3]?res1[3][0]:res0[3][0];
  assign ready[4]=c_out_4_full_n; assign c_out_4_write=active[4]; assign c_out_4_din=outbank[4]?res1[4][0]:res0[4][0];
  assign ready[5]=c_out_5_full_n; assign c_out_5_write=active[5]; assign c_out_5_din=outbank[5]?res1[5][0]:res0[5][0];
  assign ready[6]=c_out_6_full_n; assign c_out_6_write=active[6]; assign c_out_6_din=outbank[6]?res1[6][0]:res0[6][0];
  assign ready[7]=c_out_7_full_n; assign c_out_7_write=active[7]; assign c_out_7_din=outbank[7]?res1[7][0]:res0[7][0];
  generate for(p=0;p<8;p=p+1) begin:dc
    assign take[p]=active[p]&&ready[p]; assign sh0[p]=take[p]&&!outbank[p]; assign sh1[p]=take[p]&&outbank[p];
  end endgenerate

  integer r;
  always_ff @(posedge ap_clk) begin
    if(!ap_rst_n) begin
      for(r=0;r<8;r=r+1) begin active[r]<=0; outbank[r]<=0; outpos[r]<=0; pending[r]<=0; end
    end else for(r=0;r<8;r=r+1) begin
      if(!active[r]) begin
        if(finished[r][7]) begin active[r]<=1; outpos[r]<=0; end
      end else if(take[r] && outpos[r]==7) begin
        outpos[r]<=0; outbank[r]<=~outbank[r];
        if(pending[r]!=0) begin active[r]<=1; pending[r]<=pending[r]-1'b1+(finished[r][7]?1'b1:1'b0); end
        else if(finished[r][7]) begin active[r]<=1; pending[r]<=0; end
        else active[r]<=0;
      end else begin
        if(take[r]) outpos[r]<=outpos[r]+1'b1;
        if(finished[r][7]) pending[r]<=pending[r]+1'b1;
      end
    end
  end

  genvar i,j;
  generate for(i=0;i<8;i=i+1) begin:row for(j=0;j<8;j=j+1) begin:col
    if(j==7) begin:eastmost
      dut_norm_pe pe(ap_clk,ap_rst_n,ah[i][j],ahv[i][j],aff[i][j],all[i][j],abb[i][j],
       ah[i][j+1],ahv[i][j+1],aff[i][j+1],all[i][j+1],abb[i][j+1],
       bv[i][j],bvv[i][j],bv[i+1][j],bvv[i+1][j],32'sd0,32'sd0,sh0[i],sh1[i],res0[i][j],res1[i][j],finished[i][j]);
    end else begin:middle
      dut_norm_pe pe(ap_clk,ap_rst_n,ah[i][j],ahv[i][j],aff[i][j],all[i][j],abb[i][j],
       ah[i][j+1],ahv[i][j+1],aff[i][j+1],all[i][j+1],abb[i][j+1],
       bv[i][j],bvv[i][j],bv[i+1][j],bvv[i+1][j],res0[i][j+1],res1[i][j+1],sh0[i],sh1[i],res0[i][j],res1[i][j],finished[i][j]);
    end
  end end endgenerate
endmodule
