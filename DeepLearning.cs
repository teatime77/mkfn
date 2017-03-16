﻿using System;
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
        伝播先の情報
    */
    public class Propagation {
        // 伝播先の代入文
        public Reference UsedRefNorm;
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

            maxFnc = new Variable("std::max", ArgClass, null);
            minFnc = new Variable("std::min", ArgClass, null);

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
            代入文の依存関係を返す。
        */
        Dictionary<Assignment, List<Assignment>> AssignmentDependency(Variable t_var, List<Assignment> asns) {
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

                    // 代入文に右辺に含まれる変数のリスト ( 添え字にt±aを含む変数参照を除く )
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
            伝播の情報を作る。
        */
        Propagation MakePropagation(Variable t_var, Apply t_sub_1, Variable fld, Dictionary<Variable, Variable> to_delta_fld, Reference used_ref, Reference[] def_idxes) {
            Apply fld_domain = fld.Domain as Apply;

            int dim_cnt = (fld.TypeVar as ArrayType).DimCnt;

            // 変数を使用している変数参照の親の文
            Statement stmt = ParentStatement(used_ref);
            Debug.Assert(stmt is Assignment);

            Assignment asn = stmt as Assignment;

            CloneTable = new Dictionary<object, object>();
            CloneTable.Add(used_ref, null);

            asn = asn.Clone();
            used_ref = CloneTable[used_ref] as Reference;
            CloneTable = null;

            Propagation pr = new Propagation();

            List<Term> lnq_idxes = new List<Term>();

            // 左辺の変数の添え字
            Term[] left_idxes = (from t in asn.Left.Indexes select t.Clone()).ToArray();

            // 変数に項を代入するための辞書
            Dictionary<Reference, Term> subst_tbl = new Dictionary<Reference, Term>();
            Dictionary<Variable, Variable> var_tbl = new Dictionary<Variable, Variable>();

            LINQ lnq0 = null;

            // 変数参照の添え字から参照されている変数のリスト
            var used_ref_idx_vars = (from r in AllRefs(used_ref) where r != used_ref select r.VarRef);

            // 左辺の添え字が参照している変数の中で、used_refの添え字で参照されていない変数のリスト
            List<Variable> lnq_vars = (from idx in asn.Left.Indexes where idx is Reference && !used_ref_idx_vars.Contains(idx.AsReference().VarRef) select idx.AsReference().VarRef).Distinct().ToList();

            // すべての添え字に対し
            for (int dim = 0; dim < dim_cnt; dim++) {

                if (used_ref.Indexes[dim] is Reference) {
                    // 添え字が変数参照の場合

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

                        subst_tbl.Add(used_ref_idx, def_idxes[dim]);
                    }
                }
                else if (used_ref.Indexes[dim].Eq(t_sub_1)) {
                    // 添え字がt - 1の場合

                    if (def_idxes[dim].VarRef != t_var || (left_idxes[dim] as Reference).VarRef != t_var) {
                        throw new Exception();
                    }

                    // 右辺の t に t+1 を代入する。
                    subst_tbl.Add(left_idxes[dim] as Reference, Add(t_var, One()));

                    left_idxes[dim] = Add(t_var, One());

                }
                else if (used_ref.Indexes[dim].IsAdd()) {
                    // 添え字が加算の場合

                    Apply app = used_ref.Indexes[dim] as Apply;

                    if (app.Args.Length == 2 && app.Args[0] is Reference && app.Args[1] is Reference) {
                        // 引数の数が2個で、それぞれが変数参照の場合

                        Reference ref1 = app.Args[0] as Reference;
                        Reference ref2 = app.Args[1] as Reference;

                        if (ref1.VarRef.ParentVar is ForEach && ref2.VarRef.ParentVar is LINQ) {
                            // 最初がforeachのループ変数の参照で、2番目がLINQのループ変数の参照の場合

                            // 2番目の引数のLINQ
                            LINQ lnq = ref2.VarRef.ParentVar as LINQ;

                            if (lnq.Aggregate.Name == "Sum") {
                                Debug.Write("");
                            }
                            else if (lnq.Aggregate.Name == "Max") {
                                Debug.Write("");
                            }
                            else {
                                Debug.Assert(false);
                            }

                            // i
                            Variable for_var1 = (ref1).VarRef;

                            // p
                            Variable linq_var1 = (ref2).VarRef;

                            var v = from a in lnq_idxes where a.Eq(app) select a;
                            if (v.Any()) {
                                // 同じ形の添え字が処理済みの場合

                            }
                            else {
                                // 同じ形の添え字が処理済みでない場合

                                lnq0 = linq_var1.ParentVar as LINQ;

                                // ip
                                Variable for_var2 = def_idxes[dim].VarRef;

                                Apply start = Add(Add(for_var2, MaxRange(for_var1.Domain).Minus()), One());
                                Reference end = new Reference(for_var2);
                                Apply linq_var2_domain = Intersect(linq_var1.Domain, Range(start, end));

                                // p
                                Variable linq_var2 = new Variable(linq_var1.Name, linq_var1.TypeVar, linq_var2_domain);
                                var_tbl.Add(linq_var1, linq_var2);

                                lnq_vars.Add(linq_var2);

                                Debug.Assert(left_idxes[dim] is Reference);

                                // ip - p
                                Apply sub2 = Sub(for_var2, linq_var2);

                                // 右辺の dim に ip - p を代入する。
                                subst_tbl.Add(left_idxes[dim] as Reference, sub2);

                                left_idxes[dim] = sub2;

                                lnq_idxes.Add(app);
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
                else {

                    throw new Exception();
                }
            }

            pr.UsedRefNorm = new Reference(fld.Name, fld, def_idxes);

            if (lnq0 != null) {
                // LINQ の添え字がある場合

                asn.Right = TraverseRep(asn.Right,
                    delegate (object obj, out object ret) {
                        ret = obj;

                        if (obj == lnq0) {
                            // 対象のLINQの場合

                            if (lnq0.Aggregate.VarRef == MaxFnc) {
                                // 最大値の場合

                                ret = new Apply(MaxPoolFnc, lnq0.Select.Clone(var_tbl));
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

                // 右辺の i に ip - p を代入する。
                u_right = Subst(asn.Right.Clone(), subst_tbl);
            }
            else {
                // 変数の置換がない場合

                u_right = asn.Right.Clone();
            }

            Reference norm_left = new Reference(asn.Left.Name, asn.Left.VarRef, left_idxes);

            // 右辺の簡約化
            Term right_simple = SimplifyExpression(u_right.Clone());

            // δE/δu
            Variable delta_fld = to_delta_fld[asn.Left.VarRef];
            pr.DiffE = new Reference(delta_fld.Name, delta_fld, (from i in norm_left.Indexes select i.Clone()).ToArray());

            // δE/δu * δu/δx
            pr.Delta = MakeLinqMulDiff(pr, lnq_vars, Diff(norm_left.Clone(), pr.UsedRefNorm.Clone()));

            // Σ δE/δu * δ(置換右辺)/δx
            pr.DeltaSubst = MakeLinqMulDiff(pr, lnq_vars, Diff(u_right.Clone(), pr.UsedRefNorm.Clone()));

            // Σ δE/δu * δ(簡約置換右辺)/δx
            pr.DeltaSimple = MakeLinqMulDiff(pr, lnq_vars, Diff(right_simple.Clone(), pr.UsedRefNorm.Clone()));

            Term diff = SetParent(Differential(right_simple, pr.UsedRefNorm, null));

            // Σ δE/δu * 微分簡約置換右辺
            pr.RightDiff = MakeLinqMulDiff(pr, lnq_vars, diff);

            // Σ δE/δu * 簡約微分簡約置換右辺
            pr.RightDiffSimple = MakeLinqMulDiff(pr, lnq_vars, SimplifyExpression(diff.Clone()));

            // 伝播の情報を返す。
            return pr;
        }

        // u[iu, ju, k] = (from p in Range(H) from q in Range(H) select x[iu + p, ju + q] * h[p, q, k]).Sum() + b[k];
        // ix = iu + p   : 0 <= iu <= IU - 1   0 <= p <= H - 1
        //   iu = ix - p  : 0 <= ix - p <= IU - 1    ix - IU + 1 <= p <= ix  max(0, ix - IU + 1) <= p <= min(H - 1, ix)
        // jx = ju + q   : 0 <= ju <= JU - 1   0 <= q <= H - 1
        //   ju = jx - q  : 0 <= jx - q <= JU - 1    jx - JU + 1 <= q <= jx  max(0, jx - JU + 1) <= q <= min(H - 1, jx)
        // H

        /*
            コンストラクターから定義域を得る。
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
            δfld の定義式の左辺の添え字を返す。
        */
        Reference[] NormalIndexes(List<Assignment> forward_asns, Variable fld, Reference[] used_refs) {

            // δfld の定義式の左辺の添え字
            Reference[] def_idxes;

            // fldへの代入文のリストを得る。
            var def_asns = from x in forward_asns where x.Left.VarRef == fld select x;
            if (def_asns.Any()) {
                // fldへの代入文がある場合

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

                                Variable free_var = new Variable("i_" + ref_idx.Name, IntClass, new Apply(RangeFnc, init.Args[dim].Clone()));
                                free_var.ParentVar = FreeVariable;
                                def_idxes[dim] = new Reference(free_var);
                            }
                            else {
                                // LINQの変数参照の添え字でない場合

                                throw new Exception();
                            }
                        }
                    }
                    else {
                        // 変数参照の添え字がない場合

                        Variable free_var = new Variable("i_" + dim.ToString(), IntClass, new Apply(RangeFnc, init.Args[dim].Clone()));
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
        void SetFieldKind(Class cls, Variable x_var, List<Assignment> forward_asns) {
            x_var.Kind = FieldKind.CalculatedField;
            var calculated_fields = from asn in forward_asns select asn.Left.VarRef;
            foreach (Variable fld in calculated_fields) {
                fld.Kind = FieldKind.CalculatedField;
            }

            var range_vars = from x in All<Variable>(cls) where IsRange(x.Domain) from r in AllRefs(x.Domain) where r.VarRef != RangeFnc && r.VarRef.ParentVar == cls select r.VarRef;
            var domain_fields = from fld in cls.Fields where IsNew(fld.Domain) from r in AllRefs(fld.Domain) where r.VarRef != NewFnc select r.VarRef;
            
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

        public void DeepLearning() {
            Debug.WriteLine("深層学習");

            // アプリのクラスの親クラスに対し
            foreach (Class cls in Layers) {

                StringWriter sw = new StringWriter();

                //if (cls.Name != "ConvolutionalLayer" && cls.Name != "MaxPoolingLayer") continue;//????????????????????????

                Debug.WriteLine("layer : {0}", cls.Name, "");

                // コンストラクターから定義域を得る。
                SetDomainFromConstructor(cls);

                // 順伝播の関数
                Function forward = (from f in cls.Functions where f.Name == "Forward" select f).First();

                // 入力変数
                Variable x_var = (from f in cls.Fields where f.Name == "x" select f).First();
                Debug.Assert(IsNew(x_var.Domain));

                // 出力変数
                Variable y_var = (from f in cls.Fields where f.Name == "y" select f).First();
                Debug.Assert(x_var.TypeVar is ArrayType && y_var.TypeVar is ArrayType && IsNew(y_var.Domain));

                // 順伝播の関数定義の直下のforeach
                Debug.Assert(forward.BodyStatement.Statements.Count == 1 && forward.BodyStatement.Statements[0] is ForEach);
                ForEach top_for = (ForEach)forward.BodyStatement.Statements[0];

                // 時刻tの変数
                Variable t_var = null;
                if (top_for.LoopVariable.Name == "t") {
                    t_var = top_for.LoopVariable;
                }

                // すべての項のリスト
                List<Term> all_terms = new List<Term>();

                // すべての代入文のリスト
                List<Assignment> forward_asns = new List<Assignment>();
                Traverse(top_for,
                    delegate (object obj) {
                        if (obj is Term) {
                            // 項の場合

                            all_terms.Add(obj as Term);
                        }
                        else if (obj is Assignment) {
                            forward_asns.Add(obj as Assignment);
                        }
                    });

                // すべての変数参照のリスト
                Reference[] all_refs = (from t in all_terms where t is Reference select t as Reference).ToArray();

                List<Assignment> backward_asns = new List<Assignment>();

                // 変数の種類を調べます。
                SetFieldKind(cls, x_var, forward_asns);

                //------------------------------------------------------------ 順伝播
                MathJaxDelta = false;
                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>{0}順伝播</h4>", cls.Name, "");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\ \r\n", from asn in forward_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");

                // すべての代入文に対し
                foreach (Assignment asn in forward_asns) {
                    //Debug.WriteLine(asn.ToString());
                    Debug.Assert(asn.Left is Reference);

                    // 代入文の左辺の変数参照
                    Reference left = asn.Left as Reference;
                    Debug.Assert(left.Indexes != null);

                    // 左辺の変数参照の次元
                    int dim_cnt = left.Indexes.Length;

                    // 代入文の祖先のForEachのリスト
                    ForEach[] vfor = AncestorForEach(asn);
                    Debug.Assert(vfor.Length == dim_cnt);

                    // 左辺の変数参照の各添え字に対し
                    for (int dim = 0; dim < dim_cnt; dim++) {
                        if (left.Name == "a" && dim == 1) {

                            Debug.WriteLine("a[t, φ[t, n]] = (1 - u[t, φ[t, n]]) * Prod(from i in Range(n) select u[t, φ[t, i]]);");
                        }
                        else {

                            // 左辺の変数参照の添え字 = 代入文の祖先のForEachの変数
                            Debug.Assert(left.Indexes[dim] is Reference && (left.Indexes[dim] as Reference).VarRef == vfor[dim].LoopVariable);
                        }
                    }
                }

                Apply t_sub_1 = null;
                if (t_var != null) {

                    t_sub_1 = Add(new Term[] { new Reference(t_var), new Number(-1) });
                }

                Dictionary<Variable, Variable> to_delta_fld = cls.Fields.ToDictionary(fld => fld, fld => new Variable("δ" + fld.Name, fld.TypeVar, (fld.Domain == null ? null : fld.Domain.Clone())));

                // すべてのフィールドに対し
                foreach (Variable fld in cls.Fields) {

                    // フィールドの値を使用する変数参照のリスト
                    Reference[] used_refs = (from r in all_refs where r.VarRef == fld && r.Indexes != null && r.Used() select r).ToArray();

                    if (!used_refs.Any()) {
                        // フィールドの値を使用する変数参照がない場合

                        continue;
                    }

                    if(! IsNew(fld.Domain)) {
                        // フィールドの定義域がnewでない場合

                        throw new Exception();
                    }

                    //??? used_refsの中に、同じ代入文で同じ変数参照がある場合 ???

                    // δfld の定義式の左辺の添え字
                    Reference[] def_idxes = NormalIndexes(forward_asns, fld, used_refs);

                    // フィールドの値を使用する変数参照に対し、伝播の情報を作る。
                    List<Propagation> prs = (from used_ref in used_refs select MakePropagation(t_var, t_sub_1, fld, to_delta_fld, used_ref, def_idxes)).ToList();

                    Reference norm_ref = prs.First().UsedRefNorm;

                    sw.WriteLine("<hr/>");
                    sw.WriteLine("<div style='font-size:120%; color:red;'>");
                    sw.WriteLine("$$");
                    sw.WriteLine(@"\frac{{ \partial E }}{{ \partial {0} }}", MathJax(norm_ref), "");
                    sw.WriteLine("$$");
                    sw.WriteLine("</div>");

                    //------------------------------------------------------------ 順伝播先の変数の偏微分から計算式を作る。
                    sw.WriteLine("<h5>順伝播先の変数の偏微分から計算式を作る。</h5>");
                    sw.WriteLine("$$");
                    sw.Write("= ");
                    sw.WriteLine(string.Join(" + ", from pr in prs select MathJax(pr.Delta)));

                    sw.WriteLine("$$");

                    //------------------------------------------------------------  順伝播先の変数に定義式を代入する。
                    MathJaxDelta = true;
                    sw.WriteLine("<h5>順伝播先の変数に定義式を代入する。</h5>");
                    sw.WriteLine("$$");
                    sw.Write("= ");

                    sw.WriteLine(string.Join("\r\n\\\\ + ", from pr in prs select MathJax(pr.DeltaSubst)));

                    sw.WriteLine("$$");

                    //------------------------------------------------------------  (t + 1) − 1 を t に簡約化する。
                    sw.WriteLine("<h5>簡約化する。</h5>");
                    sw.WriteLine("$$");
                    sw.Write("= ");

                    sw.WriteLine(string.Join("\r\n\\\\ + ", from pr in prs select MathJax(pr.DeltaSimple)));

                    sw.WriteLine("$$");

                    //------------------------------------------------------------  微分の計算をする。
                    sw.WriteLine("<h5>微分の計算をする。</h5>");
                    sw.WriteLine("$$");
                    sw.Write("= ");

                    sw.WriteLine(string.Join("\r\n\\\\ + ", from pr in prs select MathJax(pr.RightDiff)));

                    sw.WriteLine("$$");

                    //------------------------------------------------------------  式を簡約化する。
                    sw.WriteLine("<h5>式を簡約化する。</h5>");
                    sw.WriteLine("$$");
                    sw.Write("= ");

                    sw.WriteLine(string.Join(" + ", from pr in prs select MathJax(pr.RightDiffSimple)));

                    sw.WriteLine("$$");

                    Term result = SimplifyExpression(Add((from pr in prs select pr.RightDiffSimple.Clone()).ToArray()));

                    Variable delta_fld = to_delta_fld[fld];
                    Reference left_delta = new Reference(delta_fld.Name, delta_fld, (from i in norm_ref.Indexes select i.Clone()).ToArray());
                    backward_asns.Add(new Assignment(left_delta, result));
                }

                // 型推論
                TypeInference(cls);

                foreach(Assignment asn in backward_asns) {
                    TypeInference(asn);
                }

                // フィールドのリストを保存する。
                List<Variable> sv_flds = new List<Variable>(cls.Fields);

                // フィールドのリストにデルタ変数を追加する。
                foreach(Variable delta_fld in to_delta_fld.Values) {
                    cls.AddField(delta_fld);
                }

                // ソースコードを作る。
                List<Assignment> sorted_backward_asns;
                MakeAllSourceCode(cls, x_var, y_var, t_var, to_delta_fld, forward_asns, backward_asns, out sorted_backward_asns);

                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>逆伝播</h4>");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\", from asn in sorted_backward_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");

                //  MathJaxを含むHTMLファイルを書く。
                WriteMathJax(sw, cls.Name);

                Traverse(cls,
                    delegate (object obj) {
                        if (obj is Reference) {
                            // 変数参照の場合

                            Debug.Assert((obj as Reference).VarRef != null);
                        }
                        else if (obj is Variable) {
                            // 変数の場合

                            Debug.Assert((obj as Variable).TypeVar != null);
                        }
                    });

                // フィールドのリストを復元する。
                cls.Fields = sv_flds;
            }
        }
    }
}