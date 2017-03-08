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
            sw.WriteLine("__global__ static void {0}({1}){{", kernel_name, args);

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

            // カーネル関数の本体のコードを書く。
            sw.WriteLine(StatementCode(asn, 1));

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
            cpu_sw.WriteLine("\tint blocks_z = BatchSize;");

            Apply domain = asn.Left.VarRef.Domain as Apply;

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
            順伝播/逆伝播の関数とカーネル関数とカーネル起動関数を作る。
        */
        void MakeForwardBackward(Class cls, StringWriter body_sw, bool is_forward, List<Assignment> asns, Dictionary<Assignment, List<Assignment>> depend) {
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

                // カーネル関数のソースコードを作る。
                MakeKenel(body_sw, asn, kernel_name, flds);

                // カーネル関数の起動のソースコードを作る。
                MakeStartKenel(cls, body_sw, asn, kernel_name, flds, depend);
            }

            // 順伝播/逆伝播の関数を作る。
            string fnc_name = (is_forward ? "Forward" : "Backward");
            body_sw.WriteLine("void {0}::{1}(){{", cls.Name, fnc_name);
            body_sw.WriteLine("\tcudaMemcpyToSymbol(&_BatchSize, &BatchSize, sizeof(BatchSize));");
            foreach (Assignment asn in asns) {
                body_sw.WriteLine("\tStart_{0}();", KernelName(is_forward, asn.Left.VarRef));
            }
            body_sw.WriteLine("}");
        }

        /*
          要素の数の計算式を返す。
        */
        Term ElementCountApply(Variable va) {
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
            return s.Replace("δ", "delta_").Replace("σ", "sigmoid").Replace("std::", "");
        }

        /*
        ヘッダファイルのコードを作る。
        */
        string MakeHeaderFile(Class cls, List<Variable> array_flds, Function constructor, List<Assignment> sorted_forward_asns, List<Assignment> sorted_backward_asns) {
            StringWriter header_sw = new StringWriter();

            string header = "class " + cls.Name + " : Layer {\r\npublic:\r\n" +
                string.Join("", from fld in cls.Fields select Nest(1) + FieldCode(fld)) +
                string.Join("", from fld in array_flds select "\tcudaStream_t " + StreamName(fld) + ";\r\n");

            header_sw.WriteLine(header);

            header_sw.WriteLine("\t{0};", FunctionHeader(cls, constructor, false));
            header_sw.WriteLine("\t~{0}();", cls.Name);
            header_sw.WriteLine("\tvoid Forward();");
            header_sw.WriteLine("\tvoid Backward();");
            header_sw.WriteLine("\tvoid Allocate();");
            header_sw.WriteLine("\tvoid Free();");

            foreach (Assignment asn in sorted_forward_asns) {
                header_sw.WriteLine("\tvoid Start_{0}();", KernelName(true , asn.Left.VarRef));
            }
            foreach (Assignment asn in sorted_backward_asns) {
                header_sw.WriteLine("\tvoid Start_{0}();", KernelName(false, asn.Left.VarRef));
            }

            header_sw.WriteLine("};");

            return header_sw.ToString();
        }

        /*
            逆伝播の関数を作る。
        */
        void MakeBackward(Class cls, Variable x_var, Variable y_var, Variable t_var, Function forward, List<Assignment> forward_asns, List<Assignment> backward_asns, out List<Assignment> sorted_backward_asns) {
            MkFn.LinqValue = new Dictionary<LINQ, string>();

            // 代入文の依存関係
            Dictionary<Assignment, List<Assignment>> forward_depend = AssignmentDependency(t_var, forward_asns);
            Dictionary<Assignment, List<Assignment>> backward_depend = AssignmentDependency(t_var, backward_asns);

            List<Assignment> sorted_forward_asns = SortAssignment(forward_asns, forward_depend);
            sorted_backward_asns = SortAssignment(backward_asns, backward_depend);

            OutputLanguage = Language.CUDA;

            StringWriter sw = new StringWriter();

            if (OutputLanguage == Language.CUDA) {

                sw.WriteLine("#include <cuda_runtime.h>");
                sw.WriteLine("#include <device_launch_parameters.h>");
            }

            sw.WriteLine("#include <stdio.h>");
            sw.WriteLine("#include <FLOAT.h>");
            sw.WriteLine("#include \"MkFn.h\"");
            sw.WriteLine("#include \"{0}.h\"", cls.Name);

            sw.WriteLine("__constant__ int _BatchSize;");

            List<Variable> array_flds = cls.Fields.Where(x => x.TypeVar is ArrayType).ToList();


            // コンストラクター
            Function constructor = (from f in cls.Functions where f.IsConstructor() select f).First();


            TmpCnt = 0;

            sw.WriteLine("");
            sw.WriteLine(FunctionHeader(cls, constructor, true) + "{");

            foreach (Statement stmt in constructor.BodyStatement.Statements) {
                Variable fld = (stmt as Assignment).Left.VarRef;
                if (fld.TypeVar is ArrayType) {

                }
                else {
                    sw.WriteLine(StatementCode(stmt, 1));
                }
            }

            sw.WriteLine(string.Join("", from fld in array_flds select "\tcudaStreamCreate(&" + StreamName(fld) + ");\r\n"));
            sw.WriteLine("}");


            // デストラクタ
            sw.WriteLine("");
            sw.WriteLine("{0}::~{0}(){{", cls.Name);
            sw.WriteLine("\tFree();");
            sw.WriteLine(string.Join("", from fld in array_flds select "\tcudaStreamDestroy(" + StreamName(fld) + ");\r\n"));
            sw.WriteLine("}");

            // 配列の領域の確保
            sw.WriteLine("");
            sw.WriteLine("void {0}::Allocate(){{", cls.Name);
            sw.Write(string.Join("", from fld in array_flds select string.Format("\tcudaMalloc(&{0}, BatchSize * {1}); \r\n", fld.Name, ElementCountApply(fld).Code())));
            sw.WriteLine("}");

            // 配列の領域の解放
            sw.WriteLine("");
            sw.WriteLine("void {0}::Free(){{", cls.Name);
            sw.Write(string.Join("", from fld in array_flds select string.Format("\tcudaFree({0}); \r\n", fld.Name)));
            sw.WriteLine("}");

            // 順伝播/逆伝播の関数とカーネル関数とカーネル起動関数を作る。
            MakeForwardBackward(cls, sw, true , sorted_forward_asns, forward_depend);
            MakeForwardBackward(cls, sw, false, sorted_backward_asns, backward_depend);

            string src_dir = HomeDir + "\\src\\CUDA";
            if (!Directory.Exists(src_dir)) {

                Directory.CreateDirectory(src_dir);
            }

            if (Directory.Exists(@"Z:\")) {

                //!!!!!!!!!! デバッグ環境用 !!!!!!!!!!
                src_dir = @"Z:\prj\mkfn\src\CUDA";
            }

            // ヘッダファイルのコードを作る。
            string header_code = MakeHeaderFile(cls, array_flds, constructor, sorted_forward_asns, sorted_backward_asns);
            File.WriteAllText(src_dir + "\\" + cls.Name + ".h", ASCII(header_code), Encoding.UTF8);

            // 実装のコードをファイルに書く。
            File.WriteAllText(src_dir + "\\" + cls.Name + ".cu", ASCII(sw.ToString()), Encoding.UTF8);

            MkFn.LinqValue = null;
        }
    }
}