using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace MkFn {
    /*
        伝播先の情報
    */
    public class Propagation {
        // 伝播先の代入文
        public Assignment Asn;
        public Reference UsedRefNorm;
        public Term RightDiff;
        public Term RightDiffSimple;
        public Term Delta;
        public Term DeltaSubst;
        public Term DeltaSimple;

        public Reference DiffE;

        public Propagation(Reference used_ref, Assignment asn) {
            Asn = asn;
        }
    }

    public class ForEachNode {
        public Term DomainForEach;
        public List<ForEachNode> Children = new List<ForEachNode>();

        public ForEachNode(Term domain) {
            DomainForEach = domain;
        }
    }



    public delegate void NaviAction(object self);
    public delegate bool NaviFnc(object self, out object ret);

    //------------------------------------------------------------ TProject
    public partial class MkFn {
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

            DomainFnc = new Variable("Domain", null, null);
            DiffFnc = new Variable("Diff", ArgClass, null);

            SimpleTypes.Add(VoidClass);
            SimpleTypes.Add(IntClass);
            SimpleTypes.Add(FloatClass);
            SimpleTypes.Add(DoubleClass);
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
                    Debug.WriteLine(asn.ToString());
                }
                break;

                L1:;
            }

            return processed;
        }

        /*
        変数の依存関係
        */
        Dictionary<Assignment, List<Assignment>> AssignmentDependency(Variable t_var, List<Assignment> asns) {
            Dictionary<Assignment, List<Assignment>> dct = new Dictionary<Assignment, List<Assignment>>();

            Dictionary<Variable, Assignment> var_to_asn = asns.ToDictionary(x => x.Left.VarRef, x => x);

            foreach (Assignment asn in asns) {
                Reference[] refs = EnumReference(asn.Right);

                if (dct.ContainsKey(asn)) {
                    throw new Exception();
                }
                List<Variable> depend;

                if (t_var == null) {
                    depend = (from r in refs select r.VarRef).Distinct().ToList();
                }
                else {

                    depend = (from r in refs where r.Indexes == null || ! r.Indexes.Where(i => i.IsAdd() && (i as Apply).Args[0] is Reference && ((i as Apply).Args[0] as Reference).VarRef == t_var).Any() select r.VarRef).Distinct().ToList();
                }

                if (depend.Contains(asn.Left.VarRef)) {

                    Debug.WriteLine("依存エラー : " + asn.ToString());
                    //throw new Exception();
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
        Propagation MakePropagation(Variable t_var, Apply t_sub_1, Variable fld, Dictionary<Variable, Variable> to_delta_fld, Reference used_ref) {
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

            Propagation pr = new Propagation(used_ref, asn);

            Term[] used_ref_idxes = new Term[dim_cnt];

            List<Term> lnq_idxes = new List<Term>();

            // 左辺の変数の添え字
            Term[] left_idxes = (from t in asn.Left.Indexes select t.Clone()).ToArray();

            // 変数に項を代入するための辞書
            Dictionary<Reference, Term> subst_tbl = new Dictionary<Reference, Term>();
            Dictionary<Variable, Variable> var_tbl = new Dictionary<Variable, Variable>();

            LINQ lnq0 = null;

            // 変数参照の添え字から参照されている変数のリスト
            var used_ref_idx_vars = (from r in used_ref.AllRefs() where r != used_ref select r.VarRef);

            // 左辺の添え字が参照している変数の中で、used_refの添え字で参照されていない変数のリスト
            List<Variable> lnq_vars = (from idx in asn.Left.Indexes where idx is Reference && !used_ref_idx_vars.Contains(idx.AsReference().VarRef) select idx.AsReference().VarRef).Distinct().ToList();

            // すべての添え字に対し
            for (int dim = 0; dim < dim_cnt; dim++) {

                if (used_ref.Indexes[dim] is Reference) {
                    // 添え字が変数参照の場合

                    Reference used_ref_idx = used_ref.Indexes[dim] as Reference;
                    used_ref_idxes[dim] = used_ref_idx.Clone();

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

                        //if (! var_tbl.ContainsKey(used_ref_idx.VarRef)) {
                        //    // LINQのループ変数が変換辞書に未登録の場合

                        //    // LINQのループ変数を自由変数にする。
                        //    Variable free_var = used_ref_idx.VarRef.Clone(var_tbl);
                        //    free_var.ParentVar = FreeVariable;
                        //}

                        var v = from a in lnq_idxes where a.Eq(used_ref_idx) select a;
                        if (!v.Any()) {
                            // 未処理の場合

                            lnq_idxes.Add(used_ref_idx);
                        }
                    }
                    else {

                        if (used_ref_idx.ToString() == "φ[t, n]" || used_ref_idx.ToString() == "φ[t, i]" || asn.ToString() == "a[t, φ[t, n]] = (1 - u[t, φ[t, n]]) * (from dim in Range(N) select u[t, φ[t, i]]).Prod()") {

                        }
                        else {

                            Debug.WriteLine(used_ref_idx.ToString());
                            throw new Exception();
                        }
                    }
                }
                else if (used_ref.Indexes[dim].Eq(t_sub_1)) {
                    // 添え字がt - 1の場合

                    used_ref_idxes[dim] = new Reference(t_var);

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
                            if (!v.Any()) {
                                // 同じ形の添え字が処理済みでない場合

                                lnq_idxes.Add(app);

                                lnq0 = linq_var1.ParentVar as LINQ;

                                string name = for_var1.Name + "" + linq_var1.Name;
                                Apply for_var2_domain = new Apply(new Reference(DomainFnc), new Term[] { new Reference(used_ref.VarRef), new Number(dim) });

                                // ip
                                Variable for_var2 = new Variable(name, for_var1.TypeVar, for_var2_domain);

                                used_ref_idxes[dim] = new Reference(for_var2);

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
                            }
                            else {
                                // 同じ形の添え字が処理済みの場合

                                used_ref_idxes[dim] = used_ref.Indexes[dim].Clone();
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

                Debug.Assert(used_ref_idxes[dim] != null);
            }

            pr.UsedRefNorm = new Reference(fld.Name, fld, used_ref_idxes);

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

        /*
            クラスのソースコードをファイルに書く。
        */
        void WriteClassCode(Class cls) {
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

            // Cのソースを作る。
            MakeCode mc = new MakeCode(this);
            string header, body;

            mc.ClassCode(cls, out header, out body);

            string html_dir = HomeDir + "\\src\\C";
            if (!Directory.Exists(html_dir)) {

                Directory.CreateDirectory(html_dir);
            }

            // 宣言と実装をファイルに書く。
            File.WriteAllText(html_dir + "\\" + cls.Name + ".h"  , header, Encoding.UTF8);
            File.WriteAllText(html_dir + "\\" + cls.Name + ".cpp", body, Encoding.UTF8);
        }

        // u[iu, ju, k] = (from p in Range(H) from q in Range(H) select x[iu + p, ju + q] * h[p, q, k]).Sum() + b[k];
        // ix = iu + p   : 0 <= iu <= IU - 1   0 <= p <= H - 1
        //   iu = ix - p  : 0 <= ix - p <= IU - 1    ix - IU + 1 <= p <= ix  max(0, ix - IU + 1) <= p <= min(H - 1, ix)
        // jx = ju + q   : 0 <= ju <= JU - 1   0 <= q <= H - 1
        //   ju = jx - q  : 0 <= jx - q <= JU - 1    jx - JU + 1 <= q <= jx  max(0, jx - JU + 1) <= q <= min(H - 1, jx)
        // H

        void ForEachSkeleton(Function fnc) {
            Traverse(fnc,
                delegate (object obj) {
                    if (obj is BlockStatement) {
                        // ブロック文の場合

                        BlockStatement block = obj as BlockStatement;

                        block.Statements = (from x in block.Statements where x is ForEach select x).ToList();
                    }
                });
        }

        ForEach FindForEach(BlockStatement blc, List<Variable> domains) {
            List<ForEach> for_list = (from x in blc.Statements where x is ForEach select x as ForEach).ToList();

            foreach(Variable va in domains) {
                var v = from f in for_list where f.LoopVariable.Domain.Eq(va.Domain) select f;
                if (v.Any()) {
                    domains.Remove(va);
                    if(domains.Count == 0) {
                        return v.First();
                    }

                    return FindForEach(v.First(), domains);
                }
            }

            BlockStatement current_blc = blc;
            while (domains.Any()) {
                Variable va = domains[0];
                domains.Remove(va);

                ForEach for1 = new ForEach(va, new List<Statement>());

                current_blc.AddStatement(for1);

                current_blc = for1;
            }

            return current_blc as ForEach;
        }

        Function MakeBackward(Variable x_var, Variable y_var, Function forward, List<Assignment> sorted_backward_asns) {
            Function backward_fnc = forward.Clone() as Function;
            backward_fnc.Name = "Backward";

            ForEachSkeleton(backward_fnc);

            ForEachNode root = new ForEachNode(null);

            
            foreach (Assignment asn in sorted_backward_asns) {
                List<Variable> domains = (from r in EnumReference(asn) where r.VarRef.ParentVar == FreeVariable || r.VarRef.ParentVar is ForEach select r.VarRef).Distinct().ToList();
                if (domains.Any()) {

                    ForEach for1 = FindForEach(backward_fnc.BodyStatement, domains);
                    for1.AddStatement(asn);
                }
            }

            return backward_fnc;
        }

        public void DeepLearning() {
            Debug.WriteLine("深層学習");

            // アプリのクラスの親クラスに対し
            foreach (Class cls in Layers) {

                StringWriter sw = new StringWriter();

                //if (cls.Name != "ConvolutionalLayer" && cls.Name != "MaxPoolingLayer") continue;//????????????????????????

                Debug.WriteLine("layer : {0}", cls.Name, "");

                // 順伝播の関数
                Function forward = (from f in cls.Functions where f.Name == "Forward" select f).First();

                // 入力変数
                Variable x_var = (from f in cls.Fields where f.Name == "x" select f).First();
                Debug.Assert(Term.IsNew(x_var.Domain));

                // 出力変数
                Variable y_var = (from f in cls.Fields where f.Name == "y" select f).First();
                Debug.Assert(x_var.TypeVar is ArrayType && y_var.TypeVar is ArrayType && Term.IsNew(y_var.Domain));

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

                    // フィールドの値を使用する変数参照に対し、伝播の情報を作る。
                    List<Propagation> prs = (from used_ref in used_refs select MakePropagation(t_var, t_sub_1, fld, to_delta_fld, used_ref)).ToList();

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

                Dictionary<Assignment, List<Assignment>> dct = AssignmentDependency(t_var, forward_asns);
                Dictionary<Assignment, List<Assignment>> backward_dct = AssignmentDependency(t_var, backward_asns);
                List<Assignment> sorted_backward_asns = SortAssignment(backward_asns, backward_dct);

                Function backward_fnc = MakeBackward(x_var, y_var, forward, sorted_backward_asns);

                //backward_fnc = new Function("Backward", VoidClass);
                //backward_fnc.BodyStatement = new BlockStatement((from x in sorted_backward_asns select x as Statement).ToList());

                cls.Functions.Add(backward_fnc);
                backward_fnc.ParentVar = cls;

                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>逆伝播</h4>");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\", from asn in sorted_backward_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");

                //  MathJaxを含むHTMLファイルを書く。
                WriteMathJax(sw, cls.Name);

                // 型推論
                TypeInference(cls);

                // フィールドのリストを保存する。
                List<Variable> sv_flds = new List<Variable>(cls.Fields);

                // フィールドのリストにデルタ変数を追加する。
                cls.Fields.AddRange(to_delta_fld.Values);

                // クラスのソースコードをファイルに書く。
                WriteClassCode(cls);

                // フィールドのリストを復元する。
                cls.Fields = sv_flds;
            }
        }
    }
}