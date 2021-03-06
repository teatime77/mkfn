﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MkFn {

        [ThreadStatic]
        Variable x_var;

        [ThreadStatic]
        Variable y_var;

        [ThreadStatic]
        Variable delta_x_var;

        [ThreadStatic]
        Variable delta_y_var;

        [ThreadStatic]
        public Variable t_var;

        [ThreadStatic]
        Variable T_var;

        [ThreadStatic]
        Dictionary<Variable, Variable> to_delta_fld;

        [ThreadStatic]
        List<Variable> created_flds;

        [ThreadStatic]
        List<Variable> calculated_flds;

        [ThreadStatic]
        Dictionary<Variable, Variable> max_index_flds;

        /*
            カーネル関数のソースコードを作ります。
        */
        void MakeKenel(StringWriter sw, Assignment asn, string kernel_name, List<Variable> flds) {

            // カーネル関数の引数
            string args = string.Join(", ", from x in flds select x.Code());

            if (t_var != null) {

                args += ", " + t_var.Code();
            }

            // カーネル関数のヘッダー行
            sw.WriteLine("__global__ static void {0}({1}){{", kernel_name, args);

            sw.WriteLine("\tint _batch_idx = threadIdx.x;");

            // 代入先の添え字に対し
            for (int dim1 = 0; dim1 < asn.Left.Indexes.Length; dim1++) {
                string idx = "";
                switch (asn.Left.Indexes.Length - 1 - dim1) {
                case 0:
                    idx = "blockIdx.x";
                    break;
                case 1:
                    idx = "blockIdx.y";
                    break;
                case 2:
                    idx = "blockIdx.z";
                    break;
                }

                if (! Is_t(asn.Left.Indexes[dim1])) {

                    // 代入先の添え字をthreadIdxとblockIdxから計算します。
                    sw.WriteLine("\tint {0} = {1};", asn.Left.Indexes[dim1].ToString(), idx);
                }
            }

            // カーネル関数の本体のコードを書きます。
            sw.WriteLine(StatementCode(asn, 1));

            sw.WriteLine("}");
        }

        /*
            カーネル関数の起動のソースコードを作ります。
        */
        void MakeStartKenel(Class cls, StringWriter sw, Assignment asn, string kernel_name, List<Variable> flds, Dictionary<Assignment, List<Assignment>> depend) {
            sw.WriteLine("void {0}::Start_{1}(){{", cls.Name, kernel_name);
            sw.WriteLine("\tint blocks_x = 1;");
            sw.WriteLine("\tint blocks_y = 1;");
            sw.WriteLine("\tint blocks_z = 1;");

            Apply domain = asn.Left.VarRef.Domain as Apply;

            Debug.Assert(asn.Left.Indexes.Length <= 3);
            for (int dim1 = 0; dim1 < asn.Left.Indexes.Length; dim1++) {

                string sz = domain.Args[dim1].ToString();
                string dst = "";
                switch (asn.Left.Indexes.Length - 1 - dim1) {
                case 0:
                    dst = "blocks_x";
                    break;
                case 1:
                    dst = "blocks_y";
                    break;
                case 2:
                    dst = "blocks_z";
                    break;
                }
                if (!Is_T(domain.Args[dim1])) {

                    sw.WriteLine("\t{0} = {1};", dst, sz);
                }
            }

            sw.WriteLine("\tdim3 threadsPerBlock = dim3(BatchSize);");
            sw.WriteLine("\tdim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);");

            List<Assignment> depend_asns;
            if (depend.TryGetValue(asn, out depend_asns)) {
                foreach(Assignment dep_asn in depend_asns) {

                    // 依存するカーネル関数の実行終了のイベントを待ちます。
                    sw.WriteLine("\t_chk(cudaStreamWaitEvent({0}, {1}, 0));", StreamName(asn.Left.VarRef), EventName(dep_asn.Left.VarRef));
                }
            }

            // addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
            string args = string.Join(", ", from x in flds select x.Name);
            if (t_var != null) {

                args += ", " + t_var.Name;
            }

            sw.WriteLine("\t{0}<<<blocksPerGrid, threadsPerBlock, 0, {1}>>>({2});", kernel_name, StreamName(asn.Left.VarRef), args);

            // カーネル関数の実行終了のイベントを記録します。
            sw.WriteLine("\t_chk(cudaEventRecord({0}, {1}));", EventName(asn.Left.VarRef), StreamName(asn.Left.VarRef));

            sw.WriteLine("}");
        }

        /*
            カーネル関数の名前を返します。
        */
        public static string KernelName(bool is_forward, Variable va) {
            return (is_forward ? "forward_" : "backward_") + va.Name;
        }

        /*
            ストリーム変数の名前を返します。
        */
        public static string StreamName(Variable va) {
            return "_stream_" + va.Name;
        }

        /*
            イベント変数の名前を返します。
        */
        public static string EventName(Variable va) {
            return "_event_" + va.Name;
        }

        /*
            最大値の添え字の変数の名前を返します。
        */
        public static string IndexName(Variable va) {
            return "※" + va.Name;
        }

        /*
            順伝播/逆伝播の関数とカーネル関数とカーネル起動関数を作ります。
        */
        void CudaMakeForwardBackward(Class cls, StringWriter sw, bool is_forward, List<Assignment> asns, Dictionary<Assignment, List<Assignment>> depend) {
            // すべての代入文に対し
            foreach (Assignment asn in asns) {
                if (!IsNew(asn.Left.VarRef.Domain)) {
                    throw new Exception();
                }

                // 代入文の中で参照されているフィールド
                List<Variable> flds = (from r in AllRefs(asn) where r.VarRef.IsField() select r.VarRef).ToList();


                var linqs = All<LINQ>(asn.Right).Where(x => x.Aggregate.Name == "Max");
                if (linqs.Any()) {
                    LINQ linq = linqs.First();
                    Variable idx = max_index_flds[(linq.Select as Reference).VarRef];
                    if(!flds.Contains(idx)) {
                        flds.Add(idx);
                    }
                }


                // フィールドの定義域の中で参照されているフィールド
                List<Variable> domain_vars = (from fld in flds where IsNew(fld.Domain) from r in All<Reference>(fld.Domain) where r.VarRef.ParentVar == cls select r.VarRef).ToList();

                // 代入文と定義域の中で参照されているフィールド
                flds.AddRange(domain_vars);

                // 重複を取り除きます。
                flds = flds.Distinct().ToList();

                // カーネル関数名
                string kernel_name = KernelName(is_forward, asn.Left.VarRef);

                // カーネル関数のソースコードを作ります。
                MakeKenel(sw, asn, kernel_name, flds);

                // カーネル関数の起動のソースコードを作ります。
                MakeStartKenel(cls, sw, asn, kernel_name, flds, depend);
            }

            // 順伝播/逆伝播の関数を作ります。
            string fnc_name = (is_forward ? "Forward" : "Backward");
            sw.WriteLine("void {0}::{1}(){{", cls.Name, fnc_name);
            sw.WriteLine("\t_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));");
            foreach (Assignment asn in asns) {
                sw.WriteLine("\tStart_{0}();", KernelName(is_forward, asn.Left.VarRef));
            }
            sw.WriteLine("}");
        }


        /*
            C++の順伝播/逆伝播の関数を作ります。
        */
        void CppMakeForwardBackward(Class cls, StringWriter sw, bool is_forward, List<Assignment> asns, Dictionary<Assignment, List<Assignment>> depend) {
            // 順伝播/逆伝播の関数を作ります。
            string fnc_name = (is_forward ? "Forward" : "Backward");
            sw.WriteLine("void {0}::{1}(){{", cls.Name, fnc_name);

            // すべての代入文に対し
            foreach (Assignment asn in asns) {
                if (!IsNew(asn.Left.VarRef.Domain)) {
                    throw new Exception();
                }

                sw.WriteLine("#pragma omp parallel for");

                // 代入先の添え字に対し
                for (int dim1 = 0; dim1 < asn.Left.Indexes.Length; dim1++) {
                    Reference idx = asn.Left.Indexes[dim1] as Reference;

                    if (! Is_t(idx)) {

                        if (!IsRange(idx.VarRef.Domain)) {
                            throw new Exception();
                        }
                        Apply range = idx.VarRef.Domain as Apply;

                        if (range.Args.Length == 1) {

                            sw.WriteLine("{0}for (int {1} = 0; {1} < {2}; {1}++) {{", Nest(dim1 + 1), idx.Name, range.Args[0].ToString());
                        }
                        else {

                            sw.WriteLine("{0}for (int {1} = {2}; {1} < {3}; {1}++) {{", Nest(dim1 + 1), idx.Name, range.Args[0], range.Args[1]);
                        }
                    }
                }

                int nest = asn.Left.Indexes.Length;

                // ミニバッチ内のループの始まり
                sw.WriteLine("{0}for (int _batch_idx = 0; _batch_idx < BatchSize; _batch_idx++) {{", Nest(nest + 1));

                // 代入文のコードを書きます。
                sw.WriteLine(StatementCode(asn, nest + 2));

                // ミニバッチ内のループの終わり
                sw.WriteLine("{0}}}", Nest(nest + 1));

                // 代入先の添え字に対し
                for (int dim1 = asn.Left.Indexes.Length - 1; 0 <= dim1; dim1--) {

                    if (!Is_t(asn.Left.Indexes[dim1])) {

                        sw.WriteLine("{0}}}", Nest(dim1 + 1));
                    }
                }
            }
            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数のソースコードを作ります。
        */
        void MakeUpdateParameterKenel(StringWriter sw, Apply app, string kernel_name, List<Variable> flds) {

            // カーネル関数の引数
            string args = string.Join(", ", from x in flds select x.Code() + ", " + to_delta_fld[x].Code());

            // カーネル関数のヘッダー行
            sw.WriteLine("__global__ static void {0}({1}){{", kernel_name, args);

            string idx2 = "";

            // 代入先の添え字に対し
            for (int dim1 = 0; dim1 < app.Args.Length; dim1++) {
                string idx = null;
                string sz = "";

                switch (app.Args.Length - 1 - dim1) {
                case 0:
                    idx = "threadIdx.x";
                    sz = "blockDim.x";
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

                    idx2 = "(" + idx2 + ") * " + sz + " + ";
                }
                idx2 += idx;
            }

            sw.WriteLine("\tint _idx = {0};", idx2);
            sw.WriteLine("\tint offset = _idx * _BatchSize;");

            // カーネル関数の本体のコードを書きます。
            foreach (Variable fld in flds) {
                Variable delta_fld = to_delta_fld[fld];
                sw.WriteLine("\t{");
                sw.WriteLine("\t\t{0} sum = 0;", fld.TypeVar.Name);
                sw.WriteLine("\t\tfor (int i = 0; i < _BatchSize; i++) {");
                sw.WriteLine("\t\t\tsum += {0}[offset + i];", delta_fld.Name);
                sw.WriteLine("\t\t}");
                sw.WriteLine("\t\t{0}[_idx] -= _LearningRate * sum;", fld.Name);
                sw.WriteLine("\t}");
            }

            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数の起動のソースコードを作ります。
        */
        void MakeStartUpdateParameterKenel(Class cls, StringWriter sw, Apply app, string kernel_name, int kernel_idx, List<Variable> flds) {
            sw.WriteLine("void {0}::UpdateParameter_{1}(){{", cls.Name, kernel_idx);
            sw.WriteLine("\tint threads_x = 1;");
            sw.WriteLine("\tint blocks_x = 1;");
            sw.WriteLine("\tint blocks_y = 1;");
            sw.WriteLine("\tint blocks_z = 1;");

            for (int dim1 = 0; dim1 < app.Args.Length; dim1++) {

                string sz = app.Args[dim1].ToString();
                string dst = "";
                switch (app.Args.Length - 1 - dim1) {
                case 0:
                    dst = "threads_x";
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

            sw.WriteLine("\tdim3 threadsPerBlock = dim3(threads_x);");
            sw.WriteLine("\tdim3 blocksPerGrid   = dim3(blocks_x, blocks_y, blocks_z);");

            sw.WriteLine("\t{0}<<<blocksPerGrid, threadsPerBlock>>>({1});", kernel_name, string.Join(", ", from x in flds select x.Name + ", " + to_delta_fld[x].Name));

            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数とカーネル起動関数を作ります。
        */
        void CppMakeUpdateParameter(Class cls, StringWriter sw, int kernel_idx, List<Variable> flds) {
            sw.WriteLine("");
            sw.WriteLine("void {0}::UpdateParameter_{1}(){{", cls.Name, kernel_idx);
            sw.WriteLine("\tint _count = {0};", ElementCount(flds[0]).Code());

            sw.WriteLine("#pragma omp parallel for");
            sw.WriteLine("\tfor(int _idx = 0; _idx < _count; _idx++) {");
            sw.WriteLine("\t\tint offset = _idx * BatchSize;");

            // カーネル関数の本体のコードを書きます。
            foreach (Variable fld in flds) {
                Variable delta_fld = to_delta_fld[fld];
                sw.WriteLine("\t\t{");
                sw.WriteLine("\t\t\t{0} sum = 0;", fld.TypeVar.Name);
                sw.WriteLine("\t\t\tfor (int i = 0; i < BatchSize; i++) {");
                sw.WriteLine("\t\t\t\tsum += {0}[offset + i];", delta_fld.Name);
                sw.WriteLine("\t\t\t}");
                sw.WriteLine("\t\t\t{0}[_idx] -= LearningRate * sum;", fld.Name);
                sw.WriteLine("\t\t}");
            }
            sw.WriteLine("\t}");

            sw.WriteLine("}");
        }

        /*
            パラメータ更新のカーネル関数とカーネル起動関数を作ります。
        */
        int MakeUpdateParameter(Class cls, StringWriter sw) {
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

                if(OutputLanguage == Language.CUDA) {

                    // カーネル関数名
                    string kernel_name = "UpdateParameterKernel_" + kernel_idx.ToString();

                    // パラメータ更新のカーネル関数のソースコードを作ります。
                    MakeUpdateParameterKenel(sw, app, kernel_name, flds);

                    // パラメータ更新のカーネル関数の起動のソースコードを作ります。
                    MakeStartUpdateParameterKenel(cls, sw, app, kernel_name, kernel_idx, flds);
                }
                else {

                    CppMakeUpdateParameter(cls, sw, kernel_idx, flds);
                }

                kernel_idx++;
            }

            // パラメータ更新の関数を作ります。
            sw.WriteLine("void {0}::UpdateParameter(){{", cls.Name);
            if (OutputLanguage == Language.CUDA) {

                sw.WriteLine("\t_chk(_MemcpyToSymbol(_BatchSize, BatchSize, sizeof(BatchSize)));");
                sw.WriteLine("\t_chk(_MemcpyToSymbol(_LearningRate, LearningRate, sizeof(LearningRate)));");
                sw.WriteLine("\t_chk(cudaDeviceSynchronize());");
            }

            foreach (int i in Range(dic.Keys.Count)) {

                sw.WriteLine("\tUpdateParameter_{0}();", i);
            }

            if (OutputLanguage == Language.CUDA) {
                sw.WriteLine("\t_chk(cudaDeviceSynchronize());");
            }

            sw.WriteLine("}");

            return dic.Keys.Count;
        }

        /*
          要素の数の計算式を返します。
        */
        Term ElementCount(Variable va) {
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
          各時刻の要素の数の計算式を返します。
        */
        Term TimeElementCount(Variable va) {
            if (!IsNew(va.Domain)) {
                throw new Exception();
            }

            Apply domain = va.Domain as Apply;
            if (domain.Args[0] is Reference && (domain.Args[0] as Reference).VarRef == T_var) {

                if (domain.Args.Length == 2) {

                    return domain.Args[1];
                }
                else {

                    Apply mul = Mul(domain.Args.Skip(1).Select(x => x.Clone()).ToArray());

                    return mul;
                }
            }
            else {

                if (domain.Args.Length == 1) {

                    return domain.Args[0];
                }
                else {

                    return Mul(domain.Clone().Args);
                }
            }
        }

        /*
            ASCII文字列に変換します。
        */
        string ASCII(string s) {
            return s.Replace("δ", "delta_").Replace("σ", "sigmoid").Replace("ι", "i").Replace("std::", "").Replace("※", "_idx_");
        }

        /*
            指定したフィールドの位置を返します。
        */
        string GetFieldAddress(Variable fld) {
            if (!IsNew(fld.Domain)) {
                return fld.Name;
            }

            Apply domain = fld.Domain as Apply;
            if (domain.Args[0] is Reference && (domain.Args[0] as Reference).VarRef == T_var) {

                if (domain.Args.Length == 2) {

                    return fld.Name + " + t * " + domain.Args[1].Code() + " * BatchSize";
                }
                else {

                    Apply mul = Mul(domain.Args.Skip(1).Select(x => x.Clone()).ToArray());

                    return fld.Name + " + t * " + mul.Code() + " * BatchSize";
                }
            }
            else {

                return fld.Name;
            }
        }

        /*
        ヘッダファイルのコードを作ります。
        */
        string MakeHeaderFile(Class cls, Function constructor, List<Assignment> sorted_forward_asns, List<Assignment> sorted_backward_asns, int n_parameter_update, Language output_language) {
            StringWriter sw = new StringWriter();

            string super_class = (output_language == Language.CUDA ? "LayerCuda" : "Layer");
            string header = "class DllExport " + cls.Name + " : public " + super_class + " {\r\npublic:\r\n" +
                string.Join("", from fld in cls.Fields select Nest(1) + FieldCode(fld));

            if (OutputLanguage == Language.CUDA) {

                // CUDAのストリーム変数
                header += string.Join("", from fld in calculated_flds select "\tcudaStream_t " + StreamName(fld) + ";\r\n");
 
                // CUDAのイベント変数
                header += string.Join("", from fld in calculated_flds select "\tcudaEvent_t " + EventName(fld) + ";\r\n");
            }

            sw.WriteLine(header);

            // フィールドのサイズ
            sw.WriteLine(string.Join("\r\n", from fld in cls.Fields where fld.TypeVar.DimCnt != 0 select string.Format("\tint {0}_size_[{1}];", fld.Name, fld.TypeVar.DimCnt)));
            sw.WriteLine("");

            // フィールドの情報
            sw.WriteLine("\tvirtual int GetFieldCount() override {{ return {0}; }}", cls.Fields.Count);
            sw.WriteLine("\tvirtual void GetFieldName(int field_idx, wchar_t* name) override;");
            sw.WriteLine("\tvirtual int GetFieldDimension(int field_idx) override;");
            sw.WriteLine("\tvirtual int* GetFieldSize(int field_idx) override;");
            sw.WriteLine("\tvirtual void GetFieldValue(int field_idx, void* dst) override;");
            sw.WriteLine("\tvirtual void SetFieldValue(int field_idx, void* src) override;");
            sw.WriteLine("");

            sw.WriteLine("\t{0};", FunctionHeader(cls, constructor, false));
            sw.WriteLine("\tvirtual ~{0}();", cls.Name);
            sw.WriteLine("\tvirtual void Forward() override;");
            sw.WriteLine("\tvirtual void Backward() override;");
            sw.WriteLine("\tvirtual void Allocate() override;");
            sw.WriteLine("\tvirtual void Free() override;");

            sw.WriteLine("");
            sw.WriteLine("\tvirtual void SetInput (void* src) override {{ {0} = ({1}*)src; }}", x_var.Name, x_var.TypeVar.Name);
            sw.WriteLine("\tvirtual void* GetInput(int t = 0)  override {{ return {0}; }}", GetFieldAddress(x_var));
            sw.WriteLine("\tvirtual void* GetOutput(int t = 0) override {{ return {0}; }}", GetFieldAddress(y_var));

            sw.WriteLine("\tvirtual void SetIputDelta   (void* src) override {{ {0} = ({1}*)src; }}", delta_x_var.Name, delta_x_var.TypeVar.Name);
            sw.WriteLine("\tvirtual void SetOutputDelta (void* src) override {{ {0} = ({1}*)src; }}", delta_y_var.Name, delta_y_var.TypeVar.Name);
            sw.WriteLine("\tvirtual void* GetOutputDelta(int t = 0) override {{ return {0}; }}", GetFieldAddress(delta_y_var));
            sw.WriteLine("\tvirtual void* GetInputDelta (int t = 0) override {{ return {0}; }}", GetFieldAddress(delta_x_var));

            // 入力と出力の数
            sw.WriteLine("");
            sw.WriteLine("\tvirtual int GetInputCount()  override {{ return {0}; }}", ElementCount(x_var).Code());
            sw.WriteLine("\tvirtual int GetOutputCount() override {{ return {0}; }}", ElementCount(y_var).Code());
            sw.WriteLine("\tvirtual int GetTimeCount()       override {{ return {0}; }}", (T_var != null ? T_var.Name : "0"));
            sw.WriteLine("\tvirtual int GetTimeInputCount()  override {{ return {0}; }}", TimeElementCount(x_var).Code());
            sw.WriteLine("\tvirtual int GetTimeOutputCount() override {{ return {0}; }}", TimeElementCount(y_var).Code());

            if (T_var != null) {

                sw.WriteLine("\tvirtual void SetTimeCount(int time_count) override {{ {0} = time_count; }}", T_var.Name);
            }

            if (OutputLanguage == Language.CUDA) {

                // ストリーム変数の取得と設定
                sw.WriteLine("");
                sw.WriteLine("\tvirtual void SetInputStream (cudaStream_t src) override {{ {0} = src; }}", StreamName(x_var));
                sw.WriteLine("\tvirtual cudaStream_t GetOutputStream() override {{ return {0}; }}", StreamName(y_var));

                sw.WriteLine("\tvirtual void SetOutputDeltaStream (cudaStream_t src) override {{ {0} = src; }}", StreamName(delta_y_var));
                sw.WriteLine("\tvirtual cudaStream_t GetInputDeltaStream() override {{ return {0}; }}", StreamName(delta_x_var));

                // イベント変数の取得と設定
                sw.WriteLine("");
                sw.WriteLine("\tvirtual void SetInputEvent (cudaEvent_t src) override {{ {0} = src; }}", EventName(x_var));
                sw.WriteLine("\tvirtual cudaEvent_t GetOutputEvent() override {{ return {0}; }}", EventName(y_var));

                sw.WriteLine("\tvirtual void SetOutputDeltaEvent (cudaEvent_t src) override {{ {0} = src; }}", EventName(delta_y_var));
                sw.WriteLine("\tvirtual cudaEvent_t GetInputDeltaEvent() override {{ return {0}; }}", EventName(delta_x_var));
            }

            // 順伝播の関数
            sw.WriteLine("");
            foreach (Assignment asn in sorted_forward_asns) {
                sw.WriteLine("\tvoid Start_{0}();", KernelName(true , asn.Left.VarRef));
            }

            // 逆伝播の関数
            sw.WriteLine("");
            foreach (Assignment asn in sorted_backward_asns) {
                sw.WriteLine("\tvoid Start_{0}();", KernelName(false, asn.Left.VarRef));
            }

            // パラメータ更新の関数
            sw.WriteLine("");
            sw.WriteLine("\tvirtual void UpdateParameter() override;");
            foreach (int i in Range(n_parameter_update)) {
                sw.WriteLine("\tvoid UpdateParameter_{0}();", i);
            }

            sw.WriteLine("};");

            return sw.ToString();
        }

        /*
            コンストラクタ、デストラクタ、配列の領域の確保と解放の関数を作ります。
        */
        void MakeSetup(Class cls, Function constructor, StringWriter sw) {
            // コンストラクタを作ります。
            sw.WriteLine("");
            sw.WriteLine(FunctionHeader(cls, constructor, true) + "{");
            foreach (Statement stmt in constructor.BodyStatement.Statements) {
                Variable fld = (stmt as Assignment).Left.VarRef;
                switch (fld.Kind) {
                case FieldKind.DomainField:
                    sw.WriteLine(StatementCode(stmt, 1));
                    break;

                case FieldKind.ParameterField:
                    sw.WriteLine("\tSetNormalRand({0}, {1});", fld.Name, ElementCount(fld).Code());
                    break;
                }
            }
            if(OutputLanguage == Language.CUDA) {

                // CUDAのストリーム変数を作ります。
                sw.WriteLine(string.Join("", from fld in created_flds.Intersect(calculated_flds) select "\t_chk(cudaStreamCreate(&" + StreamName(fld) + "));\r\n"));

                // CUDAのイベント変数を作ります。
                sw.WriteLine(string.Join("", from fld in created_flds.Intersect(calculated_flds) select "\t_chk(cudaEventCreate(&" + EventName(fld) + "));\r\n"));
            }
            sw.WriteLine("}");

            // デストラクタを作ります。
            sw.WriteLine("");
            sw.WriteLine("{0}::~{0}(){{", cls.Name);
            sw.WriteLine("\tFree();");
            if (OutputLanguage == Language.CUDA) {

                // CUDAのストリーム変数を削除します。
                sw.WriteLine(string.Join("", from fld in created_flds.Intersect(calculated_flds) select "\t_chk(cudaStreamDestroy(" + StreamName(fld) + "));\r\n"));

                // CUDAのイベント変数を削除します。
                sw.WriteLine(string.Join("", from fld in created_flds.Intersect(calculated_flds) select "\t_chk(cudaEventDestroy(" + EventName(fld) + "));\r\n"));
            }
            sw.WriteLine(string.Join("", from fld in created_flds where fld.Kind == FieldKind.ParameterField select "\t_Free(" + fld.Name + ");\r\n"));
            sw.WriteLine("}");

            // 配列の領域の確保を作ります。
            sw.WriteLine("");
            sw.WriteLine("void {0}::Allocate(){{", cls.Name);
            sw.Write(string.Join("", from fld in created_flds where fld.Kind != FieldKind.ParameterField select string.Format("\t_chk(_Malloc({0}, BatchSize * {1} * sizeof({2}))); \r\n", fld.Name, ElementCount(fld).Code(), fld.TypeVar.Name)));
            sw.WriteLine("}");

            // 配列の領域の解放を作ります。
            sw.WriteLine("");
            sw.WriteLine("void {0}::Free(){{", cls.Name);
            sw.Write(string.Join("", from fld in created_flds where fld.Kind != FieldKind.ParameterField select string.Format("\t_chk(_Free({0})); \r\n", fld.Name)));
            sw.WriteLine("}");
        }

        string OutputVariation(string s, Class cls, FloatPrecision float_precision, string class_suffix) {
            s = s.Replace(cls.Name, cls.Name + class_suffix);

            switch (float_precision) {
            case FloatPrecision.Half:
                s = s.Replace("float", "half");
                break;

            case FloatPrecision.Double:
                s = s.Replace("float", "double");
                break;
            }

            return s;
        }

        /*
            フィールド情報のメソッドを作ります。
        */
        void MakeFieldInfoCode(Class cls, StringWriter sw) {

            // フィールドの名前
            sw.WriteLine("");
            sw.WriteLine("void {0}::GetFieldName(int field_idx, wchar_t* name){{", cls.Name);
            sw.WriteLine("\tswitch(field_idx){");
            for(int i = 0; i < cls.Fields.Count; i++) {
                sw.WriteLine("\t\tcase {0}: wcscpy(name, L\"{1}\"); break;", i, cls.Fields[i].Name);
            }
            sw.WriteLine("\t\tdefault: name[0] = 0; break;");
            sw.WriteLine("\t}");
            sw.WriteLine("}");
            
            // フィールドの次元
            sw.WriteLine("");
            sw.WriteLine("int {0}::GetFieldDimension(int field_idx){{", cls.Name);
            sw.WriteLine("\tswitch(field_idx){");
            for (int i = 0; i < cls.Fields.Count; i++) {
                sw.WriteLine("\tcase {0}: return {1};", i, cls.Fields[i].TypeVar.DimCnt);
            }
            sw.WriteLine("\tdefault: return -1;");
            sw.WriteLine("\t}");
            sw.WriteLine("}");

            // フィールドのサイズ
            sw.WriteLine("");
            sw.WriteLine("int* {0}::GetFieldSize(int field_idx){{", cls.Name);
            sw.WriteLine("\tswitch(field_idx){");
            for (int i = 0; i < cls.Fields.Count; i++) {
                int dim_cnt = cls.Fields[i].TypeVar.DimCnt;
                if (dim_cnt == 0) {

                    sw.WriteLine("\tcase {0}: return 0;", i);
                }
                else {
                    Variable fld = cls.Fields[i];
                    if (!IsNew(fld.Domain)) {
                        throw new Exception();
                    }

                    Apply domain = (fld.Domain as Apply).Clone();

                    sw.WriteLine("\tcase {0}:",  i);
                    for (int dim_idx = 0; dim_idx < dim_cnt; dim_idx++) {

                        sw.WriteLine("\t\t{0}_size_[{1}] = {2};", fld.Name, dim_idx, domain.Args[dim_idx]);
                    }
                    sw.WriteLine("\t\treturn {0}_size_;", fld.Name);
                }
            }
            sw.WriteLine("\tdefault: return 0;");
            sw.WriteLine("\t}");
            sw.WriteLine("}");

            // フィールドの値の取得
            sw.WriteLine("");
            sw.WriteLine("void {0}::GetFieldValue(int field_idx, void* dst){{", cls.Name);
            sw.WriteLine("\tint _cnt = GetFieldElementCount(field_idx);");
            sw.WriteLine("\tswitch(field_idx){");
            for (int i = 0; i < cls.Fields.Count; i++) {
                Class tp = cls.Fields[i].TypeVar;
                if (tp is ArrayType) {

                    sw.WriteLine("\tcase {0}: memcpy(dst, {1}, _cnt * sizeof({2})); break;", i, cls.Fields[i].Name, (tp as ArrayType).ElementType);
                }
                else {

                    sw.WriteLine("\tcase {0}: memcpy(dst, &{1}, _cnt * sizeof({2})); break;", i, cls.Fields[i].Name, tp);
                }
            }
            sw.WriteLine("\t}");
            sw.WriteLine("}");

            // フィールドの値の設定
            sw.WriteLine("");
            sw.WriteLine("void {0}::SetFieldValue(int field_idx, void* src){{", cls.Name);
            sw.WriteLine("\tint _cnt = GetFieldElementCount(field_idx);");
            sw.WriteLine("\tswitch(field_idx){");
            for (int i = 0; i < cls.Fields.Count; i++) {
                Class tp = cls.Fields[i].TypeVar;
                if (tp is ArrayType) {

                    sw.WriteLine("\tcase {0}: memcpy({1}, src, _cnt * sizeof({2})); break;", i, cls.Fields[i].Name, (tp as ArrayType).ElementType);
                }
                else {

                    sw.WriteLine("\tcase {0}: memcpy(&{1}, src, _cnt * sizeof({2})); break;", i, cls.Fields[i].Name, tp);
                }
            }
            sw.WriteLine("\t}");
            sw.WriteLine("}");
        }

        /*
            ソースコードを作ります。
        */
        void MakeSourceCode(Class cls,
            Dictionary<Assignment, List<Assignment>> forward_depend, Dictionary<Assignment, List<Assignment>> backward_depend,
            List<Assignment> sorted_forward_asns, List<Assignment> sorted_backward_asns, Language output_language, FloatPrecision float_precision) {

            MkFn.LinqValue = new Dictionary<LINQ, string>();
            OutputLanguage = output_language;

            StringWriter sw = new StringWriter();

            if (OutputLanguage == Language.CUDA) {

                sw.WriteLine("#include <cuda_runtime.h>");
                sw.WriteLine("#include <device_launch_parameters.h>");
            }

            sw.WriteLine("#include <stdlib.h>");
            sw.WriteLine("#include <stdio.h>");
            sw.WriteLine("#include <FLOAT.h>");
            sw.WriteLine("#include \"MkFn.h\"");
            sw.WriteLine("#include \"../../Lib/Lib.h\"");
            if(output_language == Language.CUDA) {
                sw.WriteLine("#include \"LibCuda.h\"");
            }

            sw.WriteLine("#include \"{0}.h\"", cls.Name);


            if (OutputLanguage == Language.CUDA) {

                sw.WriteLine("__constant__ int _BatchSize;");
                sw.WriteLine("__constant__ float _LearningRate;");
            }

            // フィールド情報のメソッドを作ります。
            MakeFieldInfoCode(cls, sw);

            // コンストラクター
            Function constructor = (from f in cls.Functions where f.IsConstructor() select f).First();

            TmpCnt = 0;

            // コンストラクタ、デストラクタ、配列の領域の確保と解放の関数を作ります。
            MakeSetup(cls, constructor, sw);

            if(OutputLanguage == Language.CUDA) {

                // 順伝播/逆伝播の関数とカーネル関数とカーネル起動関数を作ります。
                CudaMakeForwardBackward(cls, sw, true, sorted_forward_asns, forward_depend);
                CudaMakeForwardBackward(cls, sw, false, sorted_backward_asns, backward_depend);
            }
            else {

                // C++の順伝播/逆伝播の関数を作ります。
                CppMakeForwardBackward(cls, sw, true, sorted_forward_asns, forward_depend);
                CppMakeForwardBackward(cls, sw, false, sorted_backward_asns, backward_depend);
            }

            // パラメータ更新のカーネル関数とカーネル起動関数を作ります。
            int n_parameter_update = MakeUpdateParameter(cls, sw);


            sw.WriteLine("");
            sw.WriteLine("extern \"C\" DllExport Layer* Make{0}({1}){{", cls.Name, string.Join(", ", from v in constructor.Params select VariableCode(v)));
            sw.WriteLine("\treturn new {0}({1});", cls.Name, string.Join(", ", constructor.Params.Select(x => x.Name)));
            sw.WriteLine("}");

            string layer_str = "";
            string class_suffix = "";

            switch (output_language) {
            case Language.CPP:
                class_suffix = "";
                layer_str = "Layer";
                break;

            case Language.CUDA:
                class_suffix = "Cuda";
                layer_str = "LayerCUDA";
                break;
            }

            switch (float_precision) {
            case FloatPrecision.Half:
                class_suffix += "H";
                break;
            case FloatPrecision.Float:
                class_suffix += "F";
                break;
            case FloatPrecision.Double:
                class_suffix += "D";
                break;
            }


            string class_name = cls.Name + class_suffix;


            string src_dir = HomeDir + "\\src\\Layer\\" + layer_str;
            if (!Directory.Exists(src_dir)) {

                Directory.CreateDirectory(src_dir);
            }

            //!!!!!!!!!! デバッグ環境用 !!!!!!!!!!
            if (Directory.Exists(@"Z:\")) {

                src_dir = @"Z:\prj\mkfn\src\" + layer_str;
            }

            // ヘッダファイルのコードを作ります。
            string header_code = MakeHeaderFile(cls, constructor, sorted_forward_asns, sorted_backward_asns, n_parameter_update, output_language);
            header_code = OutputVariation(header_code, cls, float_precision, class_suffix);
            File.WriteAllText(src_dir + "\\" + class_name + ".h", ASCII(header_code), Encoding.UTF8);

            // 実装のコードをファイルに書きます。
            string ext = (output_language == Language.CUDA ? ".cu" : ".cpp");
            string body_code = OutputVariation(sw.ToString(), cls, float_precision, class_suffix);
            File.WriteAllText(src_dir + "\\" + class_name + ext, ASCII(body_code), Encoding.UTF8);

            MkFn.LinqValue = null;
        }

        /*
            すべての言語のソースコードを作ります。
        */
        void MakeAllSourceCode(Class cls, List<Assignment> forward_asns, List<Assignment> backward_asns, out List<Assignment> sorted_backward_asns) {

            // 代入文の依存関係を求めます。
            Dictionary<Assignment, List<Assignment>> forward_depend = AssignmentDependency(forward_asns);
            Dictionary<Assignment, List<Assignment>> backward_depend = AssignmentDependency(backward_asns);

            // 代入文の依存関係を使い代入文の実行順序を決めます。
            List<Assignment> sorted_forward_asns = SortAssignment(forward_asns, forward_depend);
            sorted_backward_asns = SortAssignment(backward_asns, backward_depend);

            // CUDAのソースコードを作ります。
            MakeSourceCode(cls, forward_depend, backward_depend, sorted_forward_asns, sorted_backward_asns, Language.CUDA, FloatPrecision.Float);
            MakeSourceCode(cls, forward_depend, backward_depend, sorted_forward_asns, sorted_backward_asns, Language.CUDA, FloatPrecision.Double);

            // C++のソースコードを作ります。
            MakeSourceCode(cls, forward_depend, backward_depend, sorted_forward_asns, sorted_backward_asns, Language.CPP, FloatPrecision.Float);
            MakeSourceCode(cls, forward_depend, backward_depend, sorted_forward_asns, sorted_backward_asns, Language.CPP, FloatPrecision.Double);
        }
    }
}