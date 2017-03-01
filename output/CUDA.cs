using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MkFn {

        /*
            カーネル関数のソースコードを作る。
        */
        void MakeKenel(StringWriter sw, Assignment asn, string kernel_name, List<Variable> flds) {

            // カーネル関数の引数
            string args = string.Join(", ", from x in flds select x.Code());

            // カーネル関数のヘッダー行
            sw.WriteLine("__global__ void {0}({1}){{", kernel_name, args);

            // 代入先の添え字に対し
            for (int dim1 = 0; dim1 < asn.Left.Indexes.Length; dim1++) {
                string idx = "";
                switch (asn.Left.Indexes.Length - 1 - dim1) {
                case 0:
                    idx = "threadIdx.x";
                    break;
                case 1:
                    idx = "blockIdx.x";
                    break;
                case 2:
                    idx = "blockIdx.y";
                    break;
                case 3:
                    idx = "blockIdx.z";
                    break;
                }

                // 代入先の添え字をthreadIdxとblockIdxから計算する。
                sw.WriteLine("\tint {0} = {1};", asn.Left.Indexes[dim1].ToString(), idx);
            }

            MakeCode mc = new MakeCode(this);

            // カーネル関数の本体のコードを書く。
            sw.WriteLine(mc.StatementCode(asn, 1));

            sw.WriteLine("}");
        }

        /*
            カーネル関数の起動のソースコードを作る。
        */
        void MakeStartKenel(Class cls, StringWriter cpu_sw, Assignment asn, string kernel_name, List<Variable> flds, Dictionary<Assignment, List<Assignment>> depend) {
            cpu_sw.WriteLine("void {0}::Start_{1}(){{", cls.Name, kernel_name);
            cpu_sw.WriteLine("\tint threadsPerBlock;");
            cpu_sw.WriteLine("\tint blocks_x = 1;");
            cpu_sw.WriteLine("\tint blocks_y = 1;");
            cpu_sw.WriteLine("\tint blocks_z = 1;");

            Apply domain = asn.Left.VarRef.Domain as Apply;

            cpu_sw.WriteLine("\tint blocks_x = 1;");
            cpu_sw.WriteLine("\tint blocks_y = 1;");
            cpu_sw.WriteLine("\tint blocks_z = 1;");
            cpu_sw.WriteLine("\tint threadsPerBlock = 1;");

            for (int dim1 = 0; dim1 < asn.Left.Indexes.Length; dim1++) {

                string sz = domain.Args[dim1].ToString();
                string dst = "";
                switch (asn.Left.Indexes.Length - 1 - dim1) {
                case 0:
                    dst = "threadsPerBlock";
                    break;
                case 1:
                    dst = "blocks_x";
                    break;
                case 2:
                    dst = "blocks_y";
                    break;
                case 3:
                    dst = "blocks_z";
                    break;
                }
                cpu_sw.WriteLine("\t{0} = {1};", dst, sz);
            }

            cpu_sw.WriteLine("\tdim3 blocksPerGrid = dim3(blocks_x, blocks_y, blocks_z);");

            List<Assignment> depend_asns;
            if (depend.TryGetValue(asn, out depend_asns)) {
                foreach(Assignment dep_asn in depend_asns) {

                    cpu_sw.WriteLine("\tcudaStreamSynchronize({0});", StreamName(dep_asn.Left.VarRef));
                }
            }

            // addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
            cpu_sw.WriteLine("\t{0}<<<blocksPerGrid, threadsPerBlock>>>({1});", kernel_name, string.Join(", ", from x in flds select x.Name));

            cpu_sw.WriteLine("}");
        }

        /*
            カーネル関数の名前を返す。
        */
        string KernelName(bool is_forward, Variable va) {
            return (is_forward ? "forward_" : "backward_") + va.Name;
        }

        /*
            ストリーム変数の名前を返す。
        */
        string StreamName(Variable va) {
            return "_stream_" + va.Name;
        }

        /*
            
        */
        void MakeCUDA(Class cls, StringWriter header_sw, StringWriter body_sw, bool is_forward, List<Assignment> asns, Dictionary<Assignment, List<Assignment>> depend) {
            Dictionary<Assignment, string> kernel_names = new Dictionary<Assignment, string>();

            // すべての代入文に対し
            foreach (Assignment asn in asns) {
                if (!IsNew(asn.Left.VarRef.Domain)) {
                    throw new Exception();
                }

                // 代入文の中で参照されているフィールド
                List<Variable> flds = (from r in AllRefs(asn) where r.VarRef.IsField() select r.VarRef).ToList();

                // フィールドの定義域の中で参照されているフィールド
                List<Variable> domain_vars = (from fld in flds where IsNew(fld.Domain) from t in fld.Domain.AsApply().Args select t.AsReference().VarRef).ToList();

                // 代入文と定義域の中で参照されているフィールド
                flds.AddRange(domain_vars);

                // 重複を取り除く。
                flds = flds.Distinct().ToList();

                // カーネル関数名
                string kernel_name = KernelName(is_forward, asn.Left.VarRef);

                // 関数名を辞書に登録する。
                kernel_names.Add(asn, kernel_name);

                // カーネル関数のソースコードを作る。
                MakeKenel(body_sw, asn, kernel_name, flds);

                // カーネル関数の起動のソースコードを作る。
                MakeStartKenel(cls, body_sw, asn, kernel_name, flds, depend);
            }

            string fnc_name = (is_forward ? "Forward" : "Backward");
            body_sw.WriteLine("void {0}::{1}(){{", cls.Name, fnc_name);
            foreach (Assignment asn in asns) {
                header_sw.WriteLine("\tvoid Start_{0}();", KernelName(is_forward, asn.Left.VarRef));
                body_sw.WriteLine("\tStart_{0}();", KernelName(is_forward, asn.Left.VarRef));
            }
            body_sw.WriteLine("}");
        }

        Term SizeApply(Variable va) {
            if (!IsNew(va.Domain)) {
                throw new Exception();
            }

            Apply domain = (va.Domain as Apply).Clone();
            if(domain.Args.Length == 1) {

                return domain.Args[0];
            }
            else {

                return Mul(domain.Args);
            }
        }

        /*
            ASCII文字列に変換する。
        */
        string ASCII(string s) {
            return s.Replace("δ", "delta_").Replace("σ", "sigmoid");
        }

        /*
            CUDAのソースコードをファイルに書く。, Variable t_var, Variable x_var
        */
        void WriteCUDAClassCode(Class cls, Dictionary<Variable, Variable> to_delta_fld, List<Assignment> sorted_asns, Dictionary<Assignment, List<Assignment>> depend) {
            OutputLanguage = Language.CUDA;

            // Cのソースを作る。
            MakeCode mc = new MakeCode(this);

            StringWriter header_sw = new StringWriter();
            StringWriter body_sw = new StringWriter();

            string s = @"#include ""cuda_runtime.h""
#include ""device_launch_parameters.h""
#include <stdio.h>";

            body_sw.WriteLine(s);
            body_sw.WriteLine("#include \"{0}.h\"", cls.Name);

            var array_flds = cls.Fields.Where(x => x.TypeVar is ArrayType);


            string header = "class " + cls.Name + " {\r\npublic:\r\n" +
                string.Join("", from fld in cls.Fields select mc.Nest(1) + mc.FieldCode(fld)) +
                string.Join("", from fld in array_flds select "\tcudaStream_t " + StreamName(fld) + ";\r\n");

            header_sw.WriteLine(header);

            foreach(Variable fld in cls.Fields) {
                Variable delta_fld;
                if (to_delta_fld.TryGetValue(fld, out delta_fld)) {

                    header_sw.Write("\t" + mc.FieldCode(delta_fld));
                }
            }

            // コンストラクター
            Function constructor = (from f in cls.Functions where f.IsConstructor() select f).First();

            header_sw.WriteLine("\t{0};", mc.FunctionHeader(cls, constructor, false));
            header_sw.WriteLine("\t~{0}();", cls.Name);
            header_sw.WriteLine("\tvoid Forward();");
            header_sw.WriteLine("\tvoid Backward();");


            mc.TmpCnt = 0;

            body_sw.WriteLine("");
            body_sw.WriteLine(mc.FunctionHeader(cls, constructor, true) + "{");

            foreach(Statement stmt in constructor.BodyStatement.Statements) {
                Variable fld = (stmt as Assignment).Left.VarRef;
                if (fld.TypeVar is ArrayType) {

                    //body_sw.WriteLine("\t{0} = ({1}*)cudaMalloc({2}); \r\n", fld.Name, (fld.TypeVar as ArrayType).ElementType.Name, SizeApply(fld).Code());
                    body_sw.WriteLine("\tcudaMalloc(&{0}, {1}); \r\n", fld.Name, SizeApply(fld).Code());
                }
                else {
                    body_sw.WriteLine(mc.StatementCode(stmt, 1));
                }
            }

            body_sw.WriteLine(string.Join("", from fld in array_flds select "\tcudaStreamCreate(&" + StreamName(fld) + ");\r\n"));
            body_sw.WriteLine("}");


            // デストラクタ
            body_sw.WriteLine("");
            body_sw.WriteLine("{0}::~{0}(){{", cls.Name);
            body_sw.Write(string.Join("", from fld in array_flds select string.Format("\tcudaFree({0}); \r\n", fld.Name)));
            body_sw.WriteLine(string.Join("", from fld in array_flds select "\tcudaStreamDestroy(" + StreamName(fld) + ");\r\n"));
            body_sw.WriteLine("}");

            MakeCUDA(cls, header_sw, body_sw, false, sorted_asns, depend);
            header_sw.WriteLine("};");

            string src_dir = HomeDir + "\\src\\CUDA";
            if (!Directory.Exists(src_dir)) {

                Directory.CreateDirectory(src_dir);
            }

            //!!!!!!!!!! デバッグ環境用 !!!!!!!!!!
            src_dir = @"Z:\prj\mkfn\src\CUDA";

            // 宣言と実装をファイルに書く。
            File.WriteAllText(src_dir + "\\" + cls.Name + ".h" , ASCII(header_sw.ToString()), Encoding.UTF8);
            File.WriteAllText(src_dir + "\\" + cls.Name + ".cu", ASCII(body_sw.ToString()), Encoding.UTF8);
        }
    }
}