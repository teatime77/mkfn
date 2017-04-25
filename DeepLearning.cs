using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace MkFn {

    /*
        プログラミング言語
    */
    public enum Language {
        CS,
        CPP,
        CUDA,
        MathJax,
    }

    /*
        浮動小数点数の精度
    */
    public enum FloatPrecision {
        Half,
        Float,
        Double,
    }

    /*
        伝播先の情報
    */
    public class Propagation {
        // 伝播先の代入文
        public Term RightDiff;
        public Term RightDiffSimple;
        public Term Delta;
        public Term DeltaSubst;
        public Term DeltaSimple;

        public Reference DiffE;
    }

    public delegate void NaviAction(object self);
    public delegate bool NaviFnc(object self, out object ret);

    //------------------------------------------------------------ TProject
    public partial class MkFn {
        public const string NewName = "new";
        public Class[] Layers;
        public Class VoidClass = new Class("void");
        public Class IntClass = new Class("int");
        public Class FloatClass = new Class("float");
        public Class DoubleClass = new Class("double");
        public Class ArgClass = new Class("arg type");
        public Number Zero() { return new Number(0); }
        public Number One() { return new Number(1) ; }

        public Variable AddFnc;
        public Variable MulFnc;
        public Variable DivFnc;
        public Variable SumFnc;
        public Variable ProdFnc;
        public Variable MaxFnc;
        public Variable MaxPoolFnc;
        public Variable MaxPoolPrimeFnc;
        public Variable maxFnc;
        public Variable minFnc;

        public Variable NewFnc;
        public Variable DomainFnc;
        public Variable DiffFnc;
        public Variable EFnc;

        public Variable RangeFnc;
        public Variable σ_prime;
        public Variable tanh_prime;
        bool MathJaxDelta = false;
        object FreeVariable = new object();

        [ThreadStatic]
        public Dictionary<object, object> CloneTable;

        [ThreadStatic]
        public static Language OutputLanguage;

        [ThreadStatic]
        public static Dictionary<LINQ, string> LinqValue;

        public MkFn() {
            AddFnc = new Variable("+", ArgClass, null);
            MulFnc = new Variable("*", ArgClass, null);
            DivFnc = new Variable("/", ArgClass, null);
            SumFnc = new Variable("Sum", ArgClass, null);
            ProdFnc = new Variable("Prod", ArgClass, null);
            MaxFnc = new Variable("Max", ArgClass, null);
            MaxPoolFnc = new Variable("MaxPool", ArgClass, null);
            MaxPoolPrimeFnc = new Variable("MaxPoolPrime", ArgClass, null);

            maxFnc = new Variable("max", ArgClass, null);
            minFnc = new Variable("min", ArgClass, null);

            NewFnc = new Variable("new", null, null);
            DomainFnc = new Variable("Domain", null, null);
            DiffFnc = new Variable("Diff", ArgClass, null);

            SimpleTypes.Add(VoidClass);
            SimpleTypes.Add(IntClass);
            SimpleTypes.Add(FloatClass);
            SimpleTypes.Add(DoubleClass);
        }

        public static string ConstructorName(Class cls) {
            return cls.Name;
        }

        public int NumberTypeOrder(Class tp1) {
            Class tp2 = (tp1 is ArrayType ? (tp1 as ArrayType).ElementType : tp1);

            if(tp2 == IntClass) {
                return 1;
            }
            else if (tp2 == FloatClass) {
                return 2;
            }
            else if (tp2 == DoubleClass) {
                return 3;
            }
            else {
                throw new Exception();
            }
        }

        /*
            代入文の依存関係を使い代入文の実行順序を決めます。
        */
        List<Assignment> SortAssignment(List<Assignment> asns, Dictionary<Assignment, List<Assignment>> dct) {
            List<Assignment> pending = new List<Assignment>(asns);
            List<Assignment> processed = new List<Assignment>();

            while (pending.Any()) {
                foreach(Assignment asn in pending) {
                    List<Assignment> depend;

                    if(! dct.TryGetValue(asn, out depend) || depend.All(x => processed.Contains(x))) {
                        // 依存する代入文がないか、依存する代入文がすべて処理済みの場合

                        pending.Remove(asn);
                        processed.Add(asn);
                        goto L1;
                    }
                }

                Debug.WriteLine("相互依存の代入文");
                foreach (Assignment asn in pending) {
                    //Debug.WriteLine(asn.ToString());
                }
                break;

                L1:;
            }

            return processed;
        }

        /*
            代入文の依存関係を返します。
        */
        Dictionary<Assignment, List<Assignment>> AssignmentDependency(List<Assignment> asns) {
            // 代入文の依存関係の辞書
            Dictionary<Assignment, List<Assignment>> dct = new Dictionary<Assignment, List<Assignment>>();

            // 代入文の左辺の代入先の変数→代入文 の辞書
            Dictionary<Variable, Assignment> var_to_asn = asns.ToDictionary(x => x.Left.VarRef, x => x);

            // すべての代入文に対し
            foreach (Assignment asn in asns) {
                // 代入文に右辺に含まれる変数参照のリスト
                Reference[] refs = EnumReference(asn.Right);

                if (dct.ContainsKey(asn)) {
                    // 代入文の依存関係の辞書に含まれる場合

                    throw new Exception();
                }

                // 依存先の変数のリスト
                List<Variable> depend;

                if (t_var == null) {
                    // 時刻tの変数がある場合

                    // 代入文に右辺に含まれる変数のリスト
                    depend = (from r in refs select r.VarRef).Distinct().ToList();
                }
                else {
                    // 時刻tの変数がない場合

                    // 代入文に右辺に含まれる変数のリスト ( 添え字にt±aを含む変数参照を除きます。 )
                    depend = (from r in refs where r.Indexes == null || ! r.Indexes.Where(i => i.IsAdd() && (i as Apply).Args[0] is Reference && ((i as Apply).Args[0] as Reference).VarRef == t_var).Any() select r.VarRef).Distinct().ToList();
                }

                if (depend.Contains(asn.Left.VarRef)) {
                    // 左辺の変数が依存先の変数のリストに含まれる場合

                    throw new Exception();
                }
                else {

                    dct.Add(asn, (from v in depend where var_to_asn.ContainsKey(v) select var_to_asn[v]).ToList());
                }
            }

            return dct;
        }

        Term MakeLinqMulDiff(Propagation pr, List<Variable> lnq_vars, Term t) {
            if (lnq_vars.Any()) {
                // LINQのループ変数がある場合

                Dictionary<Variable, Variable> var_tbl = lnq_vars.ToDictionary(x => x, x => x.Clone());
                Dictionary<Reference, Term> subst_tbl = lnq_vars.ToDictionary(x => new Reference(x), x => (new Reference(var_tbl[x])) as Term);

                return new LINQ(var_tbl.Values.ToArray(), Mul(pr.DiffE.Clone(var_tbl), Subst(t, subst_tbl, var_tbl)), new Reference(SumFnc));
            }
            else {
                // LINQのループ変数がない場合

                return Mul(pr.DiffE.Clone(), t);
            }
        }

        /*
            伝播の情報を作ります。
        */
        Propagation MakePropagation(Apply t_sub_1, Variable fld, Reference used_ref, Reference def_norm_ref) {
            Reference[] def_idxes = (from x in def_norm_ref.Indexes select x as Reference).ToArray();

            Apply fld_domain = fld.Domain as Apply;

            // フィールドの次元
            int dim_cnt = (fld.TypeVar as ArrayType).DimCnt;

            // 変数を使用している変数参照の親の文
            Statement stmt = ParentStatement(used_ref);
            Debug.Assert(stmt is Assignment);

            Assignment asn = stmt as Assignment;

            CloneTable = new Dictionary<object, object>();

            // コピー情報辞書に、used_refを登録します。
            CloneTable.Add(used_ref, null);

            // asnをコピーします。
            asn = asn.Clone();

            // asnがコピーされたときの、used_refのコピーを得ます。
            used_ref = CloneTable[used_ref] as Reference;

            // コピー情報辞書をクリアします。
            CloneTable = null;

            // 伝播の情報
            Propagation pr = new Propagation();

            List<Term> lnq_idxes = new List<Term>();

            // 代入文の左辺の変数の添え字
            Term[] left_idxes = (from t in asn.Left.Indexes select t.Clone()).ToArray();

            // 変数に項を代入するための辞書
            Dictionary<Reference, Term> subst_tbl = new Dictionary<Reference, Term>();
            Dictionary<Variable, Variable> var_tbl = new Dictionary<Variable, Variable>();

            LINQ lnq0 = null;

            // 変数参照の添え字から参照されている変数のリスト
            var used_ref_idx_vars = ((from r in AllRefs(used_ref) where r != used_ref select r.VarRef)).Distinct();

            // 左辺の添え字が参照している変数の中で、used_refの添え字で参照されていない変数のリスト
            List<Variable> lnq_vars = (from idx in asn.Left.Indexes where idx is Reference && !used_ref_idx_vars.Contains(idx.AsReference().VarRef) select idx.AsReference().VarRef).Distinct().ToList();

            // すべての添え字に対し
            for (int dim = 0; dim < dim_cnt; dim++) {

                if (used_ref.Indexes[dim] is Reference) {
                    // 添え字が変数参照の場合

                    // 変数参照の添え字
                    Reference used_ref_idx = used_ref.Indexes[dim] as Reference;

                    if (used_ref_idx.VarRef.ParentVar is ForEach) {
                        // foreachのループ変数を参照する添え字の場合

                    }
                    else if (used_ref_idx.VarRef.ParentVar is LINQ) {
                        // LINQのループ変数を参照する添え字の場合

                        // 変数参照の添え字がループ変数のLINQ
                        lnq0 = used_ref_idx.VarRef.ParentVar as LINQ;
                        Debug.Assert(lnq0.Aggregate != null);

                        if (lnq0.Aggregate.Name == "Sum") {
                            Debug.Write("");
                        }
                        else if (lnq0.Aggregate.Name == "Prod") {
                            Debug.Write("");
                        }
                        else if (lnq0.Aggregate.Name == "Max") {
                            Debug.Assert(false, "未実装");
                        }
                        else {
                            Debug.Assert(false);
                        }
                    }
                    else {

                        throw new Exception();
                    }

                    if (used_ref_idx.VarRef != def_idxes[dim].VarRef) {
                        // δfld の定義式の左辺の添え字と同じでない場合

                        subst_tbl.Add(used_ref_idx, def_idxes[dim]);
                    }
                }
                else if (used_ref.Indexes[dim].Eq(t_sub_1)) {
                    // 添え字がt - 1の場合

                    if (def_idxes[dim].VarRef != t_var || (left_idxes[dim] as Reference).VarRef != t_var) {
                        throw new Exception();
                    }

                    // 右辺の t に t+1 を代入します。
                    subst_tbl.Add(left_idxes[dim] as Reference, Add(t_var, One()));

                    // 左辺の添え字を t+1 にします。
                    left_idxes[dim] = Add(t_var, One());

                }
                else if (used_ref.Indexes[dim].IsAdd()) {
                    // 添え字が加算の場合

                    Apply app = used_ref.Indexes[dim] as Apply;

                    if (app.Args.Length == 2 && app.Args[1] is Reference) {
                        // 引数の数が2個で、それぞれが変数参照の場合

                        // 加算の2番目の引数
                        Reference ref2 = app.Args[1] as Reference;

                        if (ref2.VarRef.ParentVar is LINQ) {
                            // 2番目の引数がLINQのループ変数の参照の場合

                            // 2番目の引数のLINQ
                            LINQ lnq = ref2.VarRef.ParentVar as LINQ;

                            if (lnq.Aggregate.Name != "Sum" && lnq.Aggregate.Name != "Max") {
                                // 集計関数が総和や最大値でない場合

                                throw new Exception();
                            }

                            // p
                            Variable linq_var1 = (ref2).VarRef;

                            if (! lnq_idxes.Exists(x => x.Eq(app))) {
                                // 同じ形の添え字が処理済みでない場合

                                lnq0 = linq_var1.ParentVar as LINQ;

                                // i0
                                Variable for_var2 = def_idxes[dim].VarRef;

                                if (app.Args[0] is Reference && app.Args[0].AsReference().VarRef.ParentVar is ForEach) {
                                    // 最初の引数がforeachのループ変数の参照の場合

                                    // 加算の最初の引数
                                    Reference ref1 = app.Args[0] as Reference;

                                    // i
                                    Variable for_var1 = (ref1).VarRef;

                                    Apply start = Add(Add(for_var2, MaxRange(for_var1.Domain.Clone()).Minus()), One());
                                    Reference end = new Reference(for_var2);
                                    Apply linq_var2_domain = Intersect(linq_var1.Domain, Range(start, end));

                                    // p
                                    Variable linq_var2 = new Variable(linq_var1.Name, linq_var1.TypeVar, linq_var2_domain);
                                    var_tbl.Add(linq_var1, linq_var2);

                                    lnq_vars.Add(linq_var2);

                                    Debug.Assert(left_idxes[dim] is Reference);

                                    // i0 - p
                                    Apply sub2 = Sub(for_var2, linq_var2);

                                    // 右辺の dim に i0 - p を代入します。
                                    subst_tbl.Add(left_idxes[dim] as Reference, sub2);

                                    left_idxes[dim] = sub2;

                                    lnq_idxes.Add(app);
                                }
                                else if (app.Args[0].IsMul()) {
                                    // 最初の引数が乗算の場合

                                    Apply mul = app.Args[0] as Apply;

                                    // 最初の引数はforeachのループ変数の参照
                                    Debug.Assert(mul.Args[0] is Reference && mul.Args[0].AsReference().VarRef.ParentVar is ForEach);

                                    // 乗算の最初の引数
                                    Reference ref1 = mul.Args[0] as Reference;

                                    // i
                                    Variable for_var1 = (ref1).VarRef;

                                    // 最初の引数はforeachのループ変数の参照
                                    Debug.Assert(mul.Args[1] is Reference && mul.Args[1].AsReference().VarRef.Kind == FieldKind.DomainField);

                                    Reference H = mul.Args[1] as Reference;

                                    // i0 / H
                                    Apply div = Div(for_var2, H.VarRef);

                                    // 右辺の dim に i0 / H を代入します。
                                    subst_tbl.Add(left_idxes[dim] as Reference, div);

                                    left_idxes[dim] = div;
                                }
                                else {

                                    throw new Exception();
                                }
                            }
                        }
                        else {
                            // 最初がforeachのループ変数の参照で、2番目がLINQのループ変数の参照でない場合

                            throw new Exception();
                        }
                    }
                    else {
                        throw new Exception();
                    }
                }
                else {

                    throw new Exception();
                }
            }

            if (lnq0 != null) {
                // LINQ の添え字がある場合

                asn.Right = TraverseRep(asn.Right,
                    delegate (object obj, out object ret) {
                        ret = obj;

                        if (obj == lnq0) {
                            // 対象のLINQの場合

                            if (lnq0.Aggregate.VarRef == MaxFnc) {
                                // 最大値の場合


                                Variable max_var = (lnq0.Select as Reference).VarRef;

                                Class max_idx_type = GetArrayType(IntClass, asn.Left.VarRef.TypeVar.DimCnt);

                                Apply left_domain = asn.Left.VarRef.Domain as Apply;
                                Apply max_idx_domain = new Apply(NewFnc, IntClass, left_domain.Args.Select(x => x.Clone()).ToArray());
                                Variable max_idx = new Variable(IndexName(max_var), max_idx_type, max_idx_domain, FieldKind.CalculatedField);
                                max_idx.ParentVar = fld.ParentVar;
                                max_index_flds.Add(max_var, max_idx);

                                Reference lnq_sel = lnq0.Select as Reference;
                                Debug.Assert(lnq_sel == used_ref && lnq_sel.VarRef == fld);

                                Reference lnq_sel_def_idxes = new Reference(lnq_sel.Name, lnq_sel.VarRef, def_idxes.Select(x => x.Clone()).ToArray());

                                List<Term> max_args = new List<Term>();
                                max_args.Add(lnq_sel_def_idxes);
                                max_args.Add(asn.Left.Clone(var_tbl));
                                max_args.Add(new Reference(max_idx.Name, max_idx, asn.Left.Indexes.Select(x => x.Clone(var_tbl)).ToArray()));

                                ret = new Apply(MaxPoolFnc, max_args.ToArray());
                            }
                            else if (lnq0.Aggregate.VarRef == SumFnc || lnq0.Aggregate.VarRef == ProdFnc) {
                                // 和か積の場合

                                ret = lnq0.Select.Clone(var_tbl);
                            }
                            else {

                                throw new Exception();
                            }
                            foreach(Variable va in lnq0.Variables) {
                                va.ParentVar = FreeVariable;
                            }
                            return true;
                        }

                        return false;
                    }) as Term;
            }

            Term u_right;
            if (subst_tbl.Keys.Any()) {
                // 変数の置換がある場合

                // 右辺の i に i0 - p を代入します。
                u_right = Subst(asn.Right.Clone(), subst_tbl);
            }
            else {
                // 変数の置換がない場合

                u_right = asn.Right.Clone();
            }

            Reference norm_left = new Reference(asn.Left.Name, asn.Left.VarRef, left_idxes);

            // 右辺の簡約化
            Term right_simple = SimplifyExpression(u_right.Clone());

            // δu[*]
            Variable delta_fld = to_delta_fld[asn.Left.VarRef];
            pr.DiffE = new Reference(delta_fld.Name, delta_fld, (from i in norm_left.Indexes select i.Clone()).ToArray());

            // ∂E/∂u * ∂u/∂x
            pr.Delta = MakeLinqMulDiff(pr, lnq_vars, Diff(norm_left.Clone(), def_norm_ref.Clone()));

            // Σ ∂E/∂u * ∂(置換右辺)/∂x
            pr.DeltaSubst = MakeLinqMulDiff(pr, lnq_vars, Diff(u_right.Clone(), def_norm_ref.Clone()));

            // Σ ∂E/∂u * ∂(簡約置換右辺)/∂x
            pr.DeltaSimple = MakeLinqMulDiff(pr, lnq_vars, Diff(right_simple.Clone(), def_norm_ref.Clone()));

            Term differential = SetParent(Differential(right_simple, def_norm_ref, null));

            // Σ ∂E/∂u * 微分簡約置換右辺
            pr.RightDiff = MakeLinqMulDiff(pr, lnq_vars, differential);

            // Σ ∂E/∂u * 簡約微分簡約置換右辺
            pr.RightDiffSimple = MakeLinqMulDiff(pr, lnq_vars, SimplifyExpression(differential.Clone()));

            // 伝播の情報を返します。
            return pr;
        }

        // u[i, j, k] = (from p in Range(H) from q in Range(H) select x[i + p, j + q] * h[p, q, k]).Sum() + b[k];
        // i0 = i + p   : 0 <= i <= IU - 1   0 <= p <= H - 1
        //   i = i0 - p  : 0 <= i0 - p <= IU - 1    i0 - IU + 1 <= p <= i0  max(0, i0 - IU + 1) <= p <= min(H - 1, i0)
        // i1 = j + q   : 0 <= j <= JU - 1   0 <= q <= H - 1
        //   j = i1 - q  : 0 <= i1 - q <= JU - 1    i1 - JU + 1 <= q <= i1  max(0, i1 - JU + 1) <= q <= min(H - 1, i1)
        // 

        /*
            コンストラクターから定義域を得ます。
        */
        void SetDomainFromConstructor(Class cls) {
            // コンストラクター
            Function constructor = (from f in cls.Functions where f.IsConstructor() select f).First();

            Traverse(constructor.BodyStatement,
                delegate (object obj) {
                    if (obj is Assignment) {
                        // 代入文の場合

                        Assignment asn = obj as Assignment;

                        if(asn.Left.VarRef.TypeVar is ArrayType) {
                            // 代入先が配列の場合

                            asn.Left.VarRef.Domain = asn.Right.Clone();
                            asn.Left.VarRef.Domain.Parent = asn.Left.VarRef;
                        }
                    }
                });
        }

        /*
            δfld の定義式の左辺の添え字を返します。
        */
        Reference[] NormalIndexes(List<Assignment> forward_asns, Variable fld, Reference[] used_refs) {

            // δfld の定義式の左辺の添え字
            Reference[] def_idxes;

            // fldへの代入文のリストを得ます。
            var def_asns = from x in forward_asns where x.Left.VarRef == fld select x;
            if (def_asns.Any()) {
                // fldへの代入文がある場合

                Debug.Assert(fld.Kind == FieldKind.CalculatedField);

                // 代入文は1個のはずです。
                Debug.Assert(def_asns.Count() == 1);

                // fldへの代入文の左辺
                Reference def_asn_left = def_asns.First().Left;

                if (def_asn_left.Indexes.Where(t => !(t is Reference)).Any()) {
                    // 変数参照でない添え字がある場合

                    throw new Exception();
                }

                // fldへの代入文の左辺の添え字のコピー
                def_idxes = (from i in def_asn_left.Indexes select i.Clone() as Reference).ToArray();
            }
            else {
                // fldへの代入文がない場合
                //          →fld の定義式の左辺の添え字と同じ

                Debug.Assert(fld == x_var || fld.Kind == FieldKind.ParameterField);
                Apply init = fld.Domain as Apply;

                // 添え字の配列
                def_idxes = new Reference[init.Args.Length];

                // すべての次元に対し
                for (int dim = 0; dim < init.Args.Length; dim++) {

                    // 変数参照の添え字のリスト
                    var ref_idxes = from r in used_refs where r.Indexes[dim] is Reference select r.Indexes[dim] as Reference;
                    if (ref_idxes.Any()) {
                        // 変数参照の添え字がある場合

                        // ForEachの変数参照の添え字のリスト
                        var for_idxes = ref_idxes.Where(r => r.VarRef.ParentVar is ForEach);
                        if (for_idxes.Any()) {
                            // ForEachの変数参照の添え字がある場合

                            def_idxes[dim] = for_idxes.First().Clone();
                        }
                        else {
                            // ForEachの変数参照の添え字がない場合

                            Reference ref_idx = ref_idxes.First();
                            if (ref_idx.VarRef.ParentVar is LINQ) {
                                // LINQの変数参照の添え字の場合

                                Variable free_var = new Variable("ι_" + ref_idx.Name, IntClass, new Apply(RangeFnc, init.Args[dim].Clone()));
                                free_var.ParentVar = FreeVariable;
                                def_idxes[dim] = new Reference(free_var);
                                //def_idxes[dim].SetTypeTerm();
                            }
                            else {
                                // LINQの変数参照の添え字でない場合

                                throw new Exception();
                            }
                        }
                    }
                    else {
                        // 変数参照の添え字がない場合

                        Variable free_var = new Variable("ι_" + dim.ToString(), IntClass, new Apply(RangeFnc, init.Args[dim].Clone()));
                        free_var.ParentVar = FreeVariable;
                        def_idxes[dim] = new Reference(free_var);
                    }
                }
            }

            return def_idxes;
        }

        /*
        変数の種類を調べます。
        */
        void SetFieldKind(Class cls, List<Assignment> forward_asns) {
            x_var.Kind = FieldKind.CalculatedField;
            var calculated_fields = from asn in forward_asns select asn.Left.VarRef;
            foreach (Variable fld in calculated_fields) {
                fld.Kind = FieldKind.CalculatedField;
            }

            var range_vars = from x in All<Variable>(cls) where IsRange(x.Domain) from r in AllRefs(x.Domain) where r.VarRef.ParentVar == cls select r.VarRef;
            var domain_fields = from fld in cls.Fields where IsNew(fld.Domain) from r in AllRefs(fld.Domain) where r.VarRef.ParentVar == cls select r.VarRef;
            
            foreach (Variable fld in range_vars.Union(domain_fields).Distinct()) {
                if(fld.TypeVar != IntClass) {
                    throw new Exception();
                }
                else {

                    fld.Kind = FieldKind.DomainField;
                }
            }
            foreach(Variable fld in cls.Fields.Where(x => x.TypeVar == IntClass)) {
                Debug.Assert(fld.Kind == FieldKind.DomainField);
            }

            foreach (Variable fld in cls.Fields.Where(x => x.Kind == FieldKind.Unknown)) {
                fld.Kind = FieldKind.ParameterField;
            }
        }

        /*
            伝播の情報のMathJax出力
        */
        void PropagationMathJax(StringWriter sw, List<Propagation> prs, Reference norm_ref, Term result) {
            sw.WriteLine("<hr/>");
            sw.WriteLine("<div style='font-size:120%; color:red;'>");
            sw.WriteLine("$$");
            sw.WriteLine(@"\frac{{ \partial E }}{{ \partial {0} }}", MathJax(norm_ref), "");
            sw.WriteLine("$$");
            sw.WriteLine("</div>");

            //------------------------------------------------------------ 順伝播先の変数の偏微分から計算式を作ります。
            sw.WriteLine("<h5>順伝播先の変数の偏微分から計算式を作ります。</h5>");
            sw.WriteLine("$$");
            sw.Write("= ");
            sw.WriteLine(string.Join(" + ", from pr in prs select MathJax(pr.Delta)));

            sw.WriteLine("$$");

            //------------------------------------------------------------  順伝播先の変数に定義式を代入します。
            MathJaxDelta = true;
            sw.WriteLine("<h5>順伝播先の変数に定義式を代入します。</h5>");
            sw.WriteLine("$$");
            sw.Write("= ");

            sw.WriteLine(string.Join("\r\n\\\\ + ", from pr in prs select MathJax(pr.DeltaSubst)));

            sw.WriteLine("$$");

            //------------------------------------------------------------  (t + 1) − 1 を t に簡約化します。
            sw.WriteLine("<h5>簡約化します。</h5>");
            sw.WriteLine("$$");
            sw.Write("= ");

            sw.WriteLine(string.Join("\r\n\\\\ + ", from pr in prs select MathJax(pr.DeltaSimple)));

            sw.WriteLine("$$");

            //------------------------------------------------------------  微分の計算をします。
            sw.WriteLine("<h5>微分の計算をします。</h5>");
            sw.WriteLine("$$");
            sw.Write("= ");

            sw.WriteLine(string.Join("\r\n\\\\ + ", from pr in prs select MathJax(pr.RightDiff)));

            sw.WriteLine("$$");

            //------------------------------------------------------------  式を簡約化します。
            sw.WriteLine("<h5>式を簡約化します。</h5>");
            sw.WriteLine("$$");
            sw.Write("= ");

            sw.WriteLine(MathJax(result));

            sw.WriteLine("$$");
        }

        /*
            深層学習
        */
        public void DeepLearning() {
            // アプリのクラスの親クラスに対し
            foreach (Class cls in Layers) {

                StringWriter sw = new StringWriter();

                // コンストラクターから定義域を得ます。
                SetDomainFromConstructor(cls);

                // 順伝播の関数
                Function forward = (from f in cls.Functions where f.Name == "Forward" select f).First();

                // 入力変数
                x_var = (from f in cls.Fields where f.Name == "x" select f).First();
                Debug.Assert(IsNew(x_var.Domain));

                // 出力変数
                y_var = (from f in cls.Fields where f.Name == "y" select f).First();
                Debug.Assert(x_var.TypeVar is ArrayType && y_var.TypeVar is ArrayType && IsNew(y_var.Domain));

                // 順伝播の関数定義の直下のforeach
                Debug.Assert(forward.BodyStatement.Statements.Count == 1 && forward.BodyStatement.Statements[0] is ForEach);
                ForEach top_for = (ForEach)forward.BodyStatement.Statements[0];

                // 時刻tの変数
                t_var = null;
                var t_vars = top_for.LoopVariables.Where(va => va.Name == "t");
                if (t_vars.Any()) {
                    t_var = t_vars.First();
                }

                // 時間Tのフィールド
                T_var = null;
                var T_vars = cls.Fields.Where(fld => fld.Name == "T");
                if (T_vars.Any()) {
                    T_var = T_vars.First();
                }

                // 順伝播の代入文のリスト
                List<Assignment> forward_asns = All<Assignment>(top_for).ToList();

                // すべての変数参照のリスト
                Reference[] all_refs = All<Reference>(top_for).ToArray();

                // 変数の種類を調べます。
                SetFieldKind(cls, forward_asns);

                //------------------------------------------------------------ 順伝播
                MathJaxDelta = false;
                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>{0}順伝播</h4>", cls.Name, "");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\ \r\n", from asn in forward_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");

                // 順伝播の関数内の代入文の表明
                AssertForwardAssignment(forward_asns);

                Apply t_sub_1 = null;
                if (t_var != null) {

                    t_sub_1 = Add(new Term[] { new Reference(t_var), new Number(-1) });
                }

                // δフィールドのリスト
                // DomainField以外のフィールドに対して作ります。
                to_delta_fld = cls.Fields.Where(x => x.Kind != FieldKind.DomainField).ToDictionary(fld => fld, fld => new Variable("δ" + fld.Name, fld.TypeVar, (fld.Domain == null ? null : fld.Domain.Clone()), FieldKind.CalculatedField));

                delta_x_var = to_delta_fld[x_var];

                delta_y_var = to_delta_fld[y_var];

                max_index_flds = new Dictionary<Variable, Variable>();

                // 逆伝播の代入文のリスト
                List<Assignment> backward_asns = new List<Assignment>();

                // すべてのフィールドに対し
                foreach (Variable fld in cls.Fields) {

                    // フィールドの値を使用する変数参照のリスト
                    Reference[] used_refs = (from r in all_refs where r.VarRef == fld && r.Indexes != null && r.Used() select r).ToArray();

                    if (!used_refs.Any()) {
                        // フィールドの値を使用する変数参照がない場合

                        Debug.Assert(cls.Name == "DNC" || fld.Kind == FieldKind.DomainField || fld == y_var);
                        continue;
                    }

                    if(! IsNew(fld.Domain)) {
                        // フィールドの定義域がnewでない場合

                        throw new Exception();
                    }

                    //??? used_refsの中に、同じ代入文で同じ変数参照がある場合 ???

                    // δfld の代入文の左辺の添え字
                    Reference[] def_idxes = NormalIndexes(forward_asns, fld, used_refs);

                    // δfld の代入文の左辺の変数参照
                    Reference def_norm_ref = new Reference(fld.Name, fld, def_idxes);

                    // フィールドの値を使用する変数参照に対し、伝播の情報を作ります。
                    List<Propagation> prs = (from used_ref in used_refs select MakePropagation(t_sub_1, fld, used_ref, def_norm_ref)).ToList();

                    // δfld の代入文の右辺
                    Term result = SimplifyExpression(Add((from pr in prs select pr.RightDiffSimple.Clone()).ToArray()));

                    // 伝播の情報のMathJax出力
                    PropagationMathJax(sw, prs, def_norm_ref, result);

                    Variable delta_fld = to_delta_fld[fld];
                    Reference left_delta = new Reference(delta_fld.Name, delta_fld, (from i in def_norm_ref.Indexes select i.Clone()).ToArray());
                    //left_delta.SetTypeTerm();

                    // 逆伝播の代入文
                    Assignment backward_asn = new Assignment(left_delta, result);

                    // 型推論
                    TypeInference(backward_asn);

                    // 逆伝播の代入文のリストに追加します。
                    backward_asns.Add(backward_asn);
                }

                // フィールドのリストを保存します。
                List<Variable> sv_flds = new List<Variable>(cls.Fields);

                // フィールドのリストにデルタ変数を追加します。
                foreach(Variable delta_fld in to_delta_fld.Values) {
                    cls.AddField(delta_fld);
                }

                // フィールドのリストにMaxの添え字の変数を追加します。
                foreach (Variable fld in max_index_flds.Values) {
                    cls.AddField(fld);
                }

                created_flds = cls.Fields.Where(x => x != x_var && x != delta_y_var && (x.Kind == FieldKind.CalculatedField || x.Kind == FieldKind.ParameterField)).ToList();

                calculated_flds = (cls.Fields.Where(x => x.Kind == FieldKind.CalculatedField)).ToList();

                // ソースコードを作ります。
                List<Assignment> sorted_backward_asns;
                MakeAllSourceCode(cls, forward_asns, backward_asns, out sorted_backward_asns);

                // 逆伝播の式の一覧を書きます。
                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>逆伝播</h4>");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\", from asn in sorted_backward_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");

                //  MathJaxを含むHTMLファイルを書きます。
                WriteMathJax(sw, cls.Name);

                // VarRefとTypeVarの表明
                AssertVarRefTypeVar(cls);

                // フィールドのリストを復元します。
                cls.Fields = sv_flds;
            }
        }
    }
}