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
                    idx = "threadIdx.y";
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
        void MakeStartKenel(Class cls, StringWriter sw, Assignment asn, string kernel_name, List<Variable> flds, Dictionary<Assignment, List<Assignment>> depend) {
            sw.WriteLine("void {0}::Start_{1}(){{", cls.Name, kernel_name);
            sw.WriteLine("\tint threads_y = 1;");
            sw.WriteLine("\tint blocks_x = 1;");
            sw.WriteLine("\tint blocks_y = 1;");
            sw.WriteLine("\tint blocks_z = 1;");

            Apply domain = asn.Left.VarRef.Domain as Apply;

            for (int dim1 = 0; dim1 < asn.Left.Indexes.Length; dim1++) {

                string sz = domain.Args[dim1].ToString();
                string dst = "";
                switch (asn.Left.Indexes.Length - 1 - dim1) {
                case 0:
                    dst = "threads_y";
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
                sw.WriteLine("\t{0} = {1};", dst, sz);
            }

            sw.WriteLine("\tdim3 threadsPerBlock = dim3(BatchSize, threads_y);");
            sw.WriteLine("\tdim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);");

            List<Assignment> depend_asns;
            if (depend.TryGetValue(asn, out depend_asns)) {
                foreach(Assignment dep_asn in depend_asns) {

                    sw.WriteLine("\t_chk(cudaStreamSynchronize({0}));", StreamName(dep_asn.Left.VarRef));
                }
            }

            // addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
            sw.WriteLine("\t{0}<<<blocksPerGrid, threadsPerBlock>>>({1});", kernel_name, string.Join(", ", from x in flds select x.Name));

            sw.WriteLine("}");
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
        void MakeForwardBackward(Class cls, StringWriter sw, bool is_forward, List<Assignment> asns, Dictionary<Assignment, List<Assignment>> depend) {
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
                MakeKenel(sw, asn, kernel_name, flds);

                // カーネル関数の起動のソースコードを作る。
                MakeStartKenel(cls, sw, asn, kernel_name, flds, depend);
            }

            // 順伝播/逆伝播の関数を作る。
            string fnc_name = (is_forward ? "Forward" : "Backward");
            sw.WriteLine("void {0}::{1}(){{", cls.Name, fnc_name);
            sw.WriteLine("\t_chk(cudaMemcpyToSymbol(_BatchSize, &BatchSize, sizeof(BatchSize)));");
            foreach (Assignment asn in asns) {
                sw.WriteLine("\tStart_{0}();", KernelName(is_forward, asn.Left.VarRef));
            }
            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数のソースコードを作る。
        */
        void MakeParameterUpdateKenel(StringWriter sw, Apply app, string kernel_name, List<Variable> flds, Dictionary<Variable, Variable> to_delta_fld) {

            // カーネル関数の引数
            string args = string.Join(", ", from x in flds select x.Code() + ", " + to_delta_fld[x].Code());

            // カーネル関数のヘッダー行
            sw.WriteLine("__global__ static void {0}({1}){{", kernel_name, args);

            string offset = "";

            // 代入先の添え字に対し
            for (int dim1 = 0; dim1 < app.Args.Length; dim1++) {
                string idx = null;
                string sz = "";

                switch (app.Args.Length - 1 - dim1) {
                case 0:
                    idx = "threadIdx.y";
                    sz = "blockDim.y";
                    break;
                case 1:
                    idx = "blockIdx.x";
                    sz = "gridDim.x";
                    break;
                case 2:
                    idx = "blockIdx.y";
                    sz = "gridDim.y";
                    break;
                case 3:
                    idx = "blockIdx.z";
                    sz = "gridDim.z";
                    break;
                }

                if(dim1 != 0) {

                    offset = "(" + offset + ") * " + sz + " + ";
                }
                offset += idx;
            }

            sw.WriteLine("\tint offset = {0};", offset);

            // カーネル関数の本体のコードを書く。
            foreach(Variable fld in flds) {
                Variable delta_fld = to_delta_fld[fld];
                sw.WriteLine("\t{");
                sw.WriteLine("\t\t{0} sum = 0;", fld.TypeVar.Name);
                sw.WriteLine("\t\tfor (int i = 0; i < _BatchSize; i++) {");
                sw.WriteLine("\t\t\tsum += {0}[offset + i];", delta_fld.Name);
                sw.WriteLine("\t\t}");
                sw.WriteLine("\t\t{0}[offset] += _LearningRate * sum;", fld.Name);
                sw.WriteLine("\t}");
            }

            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数の起動のソースコードを作る。
        */
        void MakeStartParameterUpdateKenel(Class cls, StringWriter sw, Apply app, string kernel_name, int kernel_idx, List<Variable> flds, Dictionary<Variable, Variable> to_delta_fld) {
            sw.WriteLine("void {0}::ParameterUpdate_{1}(){{", cls.Name, kernel_idx);
            sw.WriteLine("\tint threads_y = 1;");
            sw.WriteLine("\tint blocks_x = 1;");
            sw.WriteLine("\tint blocks_y = 1;");
            sw.WriteLine("\tint blocks_z = 1;");

            for (int dim1 = 0; dim1 < app.Args.Length; dim1++) {

                string sz = app.Args[dim1].ToString();
                string dst = "";
                switch (app.Args.Length - 1 - dim1) {
                case 0:
                    dst = "threads_y";
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
                sw.WriteLine("\t{0} = {1};", dst, sz);
            }

            sw.WriteLine("\tdim3 threadsPerBlock = dim3(BatchSize, threads_y);");
            sw.WriteLine("\tdim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);");

            sw.WriteLine("\t{0}<<<blocksPerGrid, threadsPerBlock>>>({1});", kernel_name, string.Join(", ", from x in flds select x.Name + ", " + to_delta_fld[x].Name));

            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数とカーネル起動関数を作る。
        */
        int MakeParameterUpdate(Class cls, Dictionary<Variable, Variable> to_delta_fld, StringWriter sw) {
            var param_flds = cls.Fields.Where(x => x.Kind == FieldKind.ParameterField);
            Dictionary<Apply, List<Variable>> dic = new Dictionary<Apply, List<Variable>>(new TermEqualityComparer());
            foreach (Variable fld in param_flds) {
                List<Variable> flds;

                Debug.Assert(IsNew(fld.Domain));
                if (!dic.TryGetValue(fld.Domain as Apply, out flds)) {

                    flds = new List<Variable>();
                    dic.Add(fld.Domain as Apply, flds);
                }

                flds.Add(fld);
            }

            int kernel_idx = 0;

            foreach (Apply app in dic.Keys) {
                List<Variable> flds = dic[app];
                Debug.Assert(IsNew(app));

                // カーネル関数名
                string kernel_name = "ParameterUpdateKernel_" + kernel_idx.ToString();

                // パラメータ更新のカーネル関数のソースコードを作る。
                MakeParameterUpdateKenel(sw, app, kernel_name, flds, to_delta_fld);

                // パラメータ更新のカーネル関数の起動のソースコードを作る。
                MakeStartParameterUpdateKenel(cls, sw, app, kernel_name, kernel_idx, flds, to_delta_fld);

                kernel_idx++;
            }

            // パラメータ更新の関数を作る。
            sw.WriteLine("void {0}::ParameterUpdate(){{", cls.Name);
            sw.WriteLine("\t_chk(cudaMemcpyToSymbol(_BatchSize, &BatchSize, sizeof(BatchSize)));");
            foreach (int i in Range(dic.Keys.Count)) {
                sw.WriteLine("\tParameterUpdate_{0}();", i);
            }
            sw.WriteLine("}");

            return dic.Keys.Count;
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
        string MakeHeaderFile(Class cls, Variable x_var, Variable y_var, List<Variable> array_flds, Function constructor, List<Assignment> sorted_forward_asns, List<Assignment> sorted_backward_asns, int n_parameter_update) {
            StringWriter sw = new StringWriter();

            string header = "class " + cls.Name + " : public Layer {\r\npublic:\r\n" +
                string.Join("", from fld in cls.Fields select Nest(1) + FieldCode(fld)) +
                string.Join("", from fld in array_flds select "\tcudaStream_t " + StreamName(fld) + ";\r\n");

            sw.WriteLine(header);

            sw.WriteLine("\t{0};", FunctionHeader(cls, constructor, false));
            sw.WriteLine("\t~{0}();", cls.Name);
            sw.WriteLine("\tvirtual void Forward() override;");
            sw.WriteLine("\tvirtual void Backward() override;");
            sw.WriteLine("\tvirtual void Allocate() override;");
            sw.WriteLine("\tvirtual void Free() override;");

            sw.WriteLine("\tvirtual void SetX(void* src) override {{ _chk(cudaMemcpy({0}, src, BatchSize * {1} * sizeof({2}), cudaMemcpyHostToDevice)); }}", x_var.Name, ElementCountApply(x_var).Code(), x_var.TypeVar.Name);
            sw.WriteLine("\tvirtual void SetY(void* src) override {{ _chk(cudaMemcpy({0}, src, BatchSize * {1} * sizeof({2}), cudaMemcpyHostToDevice)); }}", y_var.Name, ElementCountApply(y_var).Code(), y_var.TypeVar.Name);

            foreach (Assignment asn in sorted_forward_asns) {
                sw.WriteLine("\tvoid Start_{0}();", KernelName(true , asn.Left.VarRef));
            }
            foreach (Assignment asn in sorted_backward_asns) {
                sw.WriteLine("\tvoid Start_{0}();", KernelName(false, asn.Left.VarRef));
            }

            // パラメータ更新の関数
            sw.WriteLine("\tvoid ParameterUpdate();");
            foreach (int i in Range(n_parameter_update)) {
                sw.WriteLine("\tvoid ParameterUpdate_{0}();", i);
            }

            sw.WriteLine("};");

            return sw.ToString();
        }

        /*
            コンストラクタ、デストラクタ、配列の領域の確保と解放の関数を作る。
        */
        void MakeSetup(Class cls, Function constructor, List<Variable> array_flds, StringWriter sw) {
            // コンストラクタを作る。
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
            sw.WriteLine(string.Join("", from fld in array_flds select "\t_chk(cudaStreamCreate(&" + StreamName(fld) + "));\r\n"));
            sw.WriteLine("}");

            // デストラクタを作る。
            sw.WriteLine("");
            sw.WriteLine("{0}::~{0}(){{", cls.Name);
            sw.WriteLine("\tFree();");
            sw.WriteLine(string.Join("", from fld in array_flds select "\t_chk(cudaStreamDestroy(" + StreamName(fld) + "));\r\n"));
            sw.WriteLine("}");

            // 配列の領域の確保を作る。
            sw.WriteLine("");
            sw.WriteLine("void {0}::Allocate(){{", cls.Name);
            sw.Write(string.Join("", from fld in array_flds select string.Format("\t_chk(cudaMalloc(&{0}, BatchSize * {1} * sizeof({2}))); \r\n", fld.Name, ElementCountApply(fld).Code(), fld.TypeVar.Name)));
            sw.WriteLine("}");

            // 配列の領域の解放を作る。
            sw.WriteLine("");
            sw.WriteLine("void {0}::Free(){{", cls.Name);
            sw.Write(string.Join("", from fld in array_flds select string.Format("\t_chk(cudaFree({0})); \r\n", fld.Name)));
            sw.WriteLine("}");
        }

        /*
            ソースコードを作る。
        */
        void MakeSourceCode(Class cls, Variable x_var, Variable y_var, Variable t_var, Dictionary<Variable, Variable> to_delta_fld,
            Dictionary<Assignment, List<Assignment>> forward_depend, Dictionary<Assignment, List<Assignment>> backward_depend,
            List<Assignment> sorted_forward_asns, List<Assignment> sorted_backward_asns, Language output_language) {

            MkFn.LinqValue = new Dictionary<LINQ, string>();
            OutputLanguage = output_language;

            StringWriter sw = new StringWriter();

            if (OutputLanguage == Language.CUDA) {

                sw.WriteLine("#include <cuda_runtime.h>");
                sw.WriteLine("#include <device_launch_parameters.h>");
            }

            sw.WriteLine("#include <stdio.h>");
            sw.WriteLine("#include <FLOAT.h>");
            sw.WriteLine("#include \"../Lib/Lib.h\"");
            sw.WriteLine("#include \"MkFn.h\"");
            sw.WriteLine("#include \"{0}.h\"", cls.Name);

            sw.WriteLine("__constant__ int _BatchSize;");
            sw.WriteLine("__constant__ float _LearningRate;");

            List<Variable> array_flds = cls.Fields.Where(x => x.TypeVar is ArrayType).ToList();


            // コンストラクター
            Function constructor = (from f in cls.Functions where f.IsConstructor() select f).First();

            TmpCnt = 0;

            // コンストラクタ、デストラクタ、配列の領域の確保と解放の関数を作る。
            MakeSetup(cls, constructor, array_flds, sw);

            // 順伝播/逆伝播の関数とカーネル関数とカーネル起動関数を作る。
            MakeForwardBackward(cls, sw, true, sorted_forward_asns, forward_depend);
            MakeForwardBackward(cls, sw, false, sorted_backward_asns, backward_depend);

            // パラメータ更新のカーネル関数とカーネル起動関数を作る。
            int n_parameter_update = MakeParameterUpdate(cls, to_delta_fld, sw);

            string lang_str = output_language.ToString();
            lang_str = lang_str.Substring(lang_str.IndexOf('.') + 1);
            string src_dir = HomeDir + "\\src\\" + lang_str;
            if (!Directory.Exists(src_dir)) {

                Directory.CreateDirectory(src_dir);
            }

            //!!!!!!!!!! デバッグ環境用 !!!!!!!!!!
            if (output_language == Language.CUDA && Directory.Exists(@"Z:\")) {

                src_dir = @"Z:\prj\mkfn\src\" + lang_str;
            }

            // ヘッダファイルのコードを作る。
            string header_code = MakeHeaderFile(cls, x_var, y_var, array_flds, constructor, sorted_forward_asns, sorted_backward_asns, n_parameter_update);
            File.WriteAllText(src_dir + "\\" + cls.Name + ".h", ASCII(header_code), Encoding.UTF8);

            // 実装のコードをファイルに書く。
            string ext = (output_language == Language.CUDA ? ".cu" : ".cpp");
            File.WriteAllText(src_dir + "\\" + cls.Name + ext, ASCII(sw.ToString()), Encoding.UTF8);

            MkFn.LinqValue = null;
        }

        /*
            すべての言語のソースコードを作る。
        */
        void MakeAllSourceCode(Class cls, Variable x_var, Variable y_var, Variable t_var, Dictionary<Variable, Variable> to_delta_fld, List<Assignment> forward_asns, List<Assignment> backward_asns, out List<Assignment> sorted_backward_asns) {

            // 代入文の依存関係
            Dictionary<Assignment, List<Assignment>> forward_depend = AssignmentDependency(t_var, forward_asns);
            Dictionary<Assignment, List<Assignment>> backward_depend = AssignmentDependency(t_var, backward_asns);

            List<Assignment> sorted_forward_asns = SortAssignment(forward_asns, forward_depend);
            sorted_backward_asns = SortAssignment(backward_asns, backward_depend);

            // CUDAのソースコードを作る。
            MakeSourceCode(cls, x_var, y_var, t_var, to_delta_fld, forward_depend, backward_depend, sorted_forward_asns, sorted_backward_asns, Language.CUDA);

            // C++のソースコードを作る。
            MakeSourceCode(cls, x_var, y_var, t_var, to_delta_fld, forward_depend, backward_depend, sorted_forward_asns, sorted_backward_asns, Language.CPP);
        }
    }
}