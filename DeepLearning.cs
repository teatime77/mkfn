using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public enum IndexType {
        Simple,
        PrevT,
        Linq,
        PlusLinq
    }

    public class Propagation {
        public Assignment Asn;
        public Reference UsedRef;
        public IndexType[] IndexTypes;

        public Reference NormRef;
        public Term RightDiff;
        public Term RightDiffSimple;
        public Term Delta;
        public Term DeltaSubst;
        public Term DeltaSimple;

        public Apply DiffE;

        public Propagation(Reference used_ref, Assignment asn) {
            UsedRef = used_ref;
            IndexTypes = new IndexType[used_ref.Indexes.Length];
            Asn = asn;
        }
    }

    public delegate void NaviAction(object self);
    public delegate bool NaviFnc(object self, out object ret);

    //------------------------------------------------------------ TProject
    public partial class MkFn {
        public Class[] Layers;
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

        public void DeepLearning() {
            Debug.WriteLine("深層学習");

            // アプリのクラスの親クラスに対し
            foreach (Class cls in Layers) {

                StringWriter sw = new StringWriter();

                //if (cls.Name != "ConvolutionalLayer" && cls.Name != "MaxPoolingLayer") continue;//????????????????????????

                Debug.WriteLine("layer : {0}", cls.Name, "");

                // 順伝播の関数
                Function fnc = (from f in cls.Functions where f.Name == "Forward" select f).First();

                // 入力変数
                Variable x_var = (from f in cls.Fields where f.Name == "x" select f).First();

                // 出力変数
                Variable y_var = (from f in cls.Fields where f.Name == "y" select f).First();
                Debug.Assert(x_var.TypeVar is ArrayType && y_var.TypeVar is ArrayType);

                // 順伝播の関数定義の直下のforeach
                Debug.Assert(fnc.BodyStatement.Statements.Count == 1 && fnc.BodyStatement.Statements[0] is ForEach);
                ForEach top_for = (ForEach)fnc.BodyStatement.Statements[0];

                // 時刻tの変数
                Variable t_var = null;
                if(top_for.LoopVariable.Name == "t") {
                    t_var = top_for.LoopVariable;
                }

                // すべての項のリスト
                List<Term> all_terms = new List<Term>();

                // すべての代入文のリスト
                List<Assignment> all_asns = new List<Assignment>();
                Navi(top_for, 
                    delegate (object obj) {
                        if(obj is Term) {
                            all_terms.Add(obj as Term);
                        }
                        else if (obj is Assignment) {
                            all_asns.Add(obj as Assignment);
                        }
                    });

                // すべての変数参照のリスト
                Reference[] all_refs = (from t in all_terms where t is Reference select t as Reference).ToArray();

                //------------------------------------------------------------ 順伝播
                MathJaxDelta = false;
                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>{0}順伝播</h4>", cls.Name, "");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\ \r\n", from asn in all_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");

                // すべての代入文に対し
                foreach (Assignment asn in all_asns) {
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
                if(t_var != null) {

                    t_sub_1 = Add(new Term[] { new Reference(t_var), new Number(-1) });
                }

                // すべてのフィールドに対し
                foreach (Variable fld in cls.Fields) {

                    // 左辺の変数参照の補正値と伝播の辞書
                    List<Propagation> Props = new List<Propagation>();

                    // フィールドの値を使用する変数参照のリスト
                    Reference[] used_refs = (from r in all_refs where r.VarRef == fld && r.Indexes != null && r.Used() select r).ToArray();

                    if (! used_refs.Any()) {
                        // フィールドの値を使用する変数参照がない場合

                        continue;
                    }

                    // フィールドの値を使用する変数参照に対し伝播を作る。
                    foreach (Reference used_ref in used_refs) {
                        Propagation pr = MakePropagation(t_var, t_sub_1, fld, used_ref);
                        Props.Add(pr);
                    }

                    List<Propagation> prs = Props;
                    Reference norm_ref = prs.First().NormRef;

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

                    Function field_calc = new Function(fld.Name, fld.TypeVar);
                    Return rtn_stmt = new Return(result);

                    field_calc.BodyStatement = new BlockStatement(new List<Statement>() { rtn_stmt });
                    cls.Functions.Add(field_calc);
                    field_calc.ParentVar = cls;


                    sw.WriteLine("<pre><b>");
                    sw.WriteLine("double δ_" + norm_ref.Name + "(" + string.Join(", ", from i in norm_ref.Indexes select "int " + i.ToString()) + "){");
                    sw.WriteLine("\treturn " + result.ToString() + ";");
                    sw.WriteLine("}");
                    sw.WriteLine("</b></pre>");
                }

                WriteMathJax(sw, cls.Name);
            }

            // 型推論
            TypeInference();

            foreach (Class cls in Layers) {
                WriteClassCode(cls);
            }
        }


        Term MakeLinqMulDiff(Propagation pr, List<Variable> lnq_vars, Term t) {
            if (lnq_vars.Any()) {
                // LINQのループ変数がある場合

                return new LINQ(lnq_vars.ToArray(), Mul(pr.DiffE.Clone(),t), new Reference(SumFnc));
            }
            else {
                // LINQのループ変数がない場合

                return Mul(pr.DiffE.Clone(), t);
            }
        }

        /*
            伝播の情報を作る。
        */
        Propagation MakePropagation(Variable t_var, Apply t_sub_1, Variable fld, Reference used_ref) {
            int dim_cnt = (fld.TypeVar as ArrayType).DimCnt;

            // 変数を使用している変数参照の親の文
            Statement stmt = ParentStatement(used_ref);
            Debug.Assert(stmt is Assignment);

            Assignment asn = stmt as Assignment;
            Propagation pr = new Propagation(used_ref, asn);

            Term[] x_idxes = new Term[dim_cnt];
            List<int> plus_linq_dim = new List<int>();


            List<Term> lnq_idxes = new List<Term>();
            Term[] u_idxes = (from t in asn.Left.Indexes select t.Clone()).ToArray();
            Dictionary<Reference, Term> subst_tbl = new Dictionary<Reference, Term>();
            Dictionary<Variable, Variable> var_tbl = new Dictionary<Variable, Variable>();

            LINQ lnq0 = null;

            // 変数参照の添え字から参照されている変数のリスト
            var all_vars = (from r in used_ref.AllRefs() where r != used_ref select r.VarRef);

            // 左辺の添え字が参照している変数の中で、used_refの添え字で参照されていない変数のリスト
            List<Variable> lnq_vars = (from idx in asn.Left.Indexes where idx is Reference && !all_vars.Contains(idx.AsReference().VarRef) select idx.AsReference().VarRef).ToList();

            // すべての添え字に対し
            for (int dim = 0; dim < dim_cnt; dim++) {

                if (used_ref.Indexes[dim] is Reference) {
                    // 添え字が変数参照の場合

                    Reference idx_ref = used_ref.Indexes[dim] as Reference;
                    x_idxes[dim] = idx_ref.Clone();

                    if (idx_ref.VarRef.ParentVar is ForEach) {
                        // foreachのループ変数を参照する添え字の場合

                        pr.IndexTypes[dim] = IndexType.Simple;
                    }
                    else if (idx_ref.VarRef.ParentVar is LINQ) {
                        // LINQのループ変数を参照する添え字の場合

                        pr.IndexTypes[dim] = IndexType.Linq;

                        // 変数参照の添え字がループ変数のLINQ
                        lnq0 = idx_ref.VarRef.ParentVar as LINQ;
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

                        var v = from a in lnq_idxes where a.Eq(idx_ref) select a;
                        if (!v.Any()) {
                            // 未処理の場合

                            lnq_idxes.Add(idx_ref);
                        }
                    }
                    else {

                        if (idx_ref.ToString() == "φ[t, n]" || idx_ref.ToString() == "φ[t, i]" || stmt.ToString() == "a[t, φ[t, n]] = (1 - u[t, φ[t, n]]) * (from dim in Range(N) select u[t, φ[t, i]]).Prod()") {

                        }
                        else {

                            Debug.WriteLine(idx_ref.ToString());
                            throw new Exception();
                        }
                    }
                }
                else if (used_ref.Indexes[dim].Eq(t_sub_1)) {
                    // 添え字がt - 1の場合

                    pr.IndexTypes[dim] = IndexType.PrevT;
                    x_idxes[dim] = new Reference(t_var);
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

                            pr.IndexTypes[dim] = IndexType.PlusLinq;

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

                            plus_linq_dim.Add(dim);

                            var v = from a in lnq_idxes where a.Eq(app) select a;
                            if (!v.Any()) {
                                // 同じ形の添え字が処理済みでない場合

                                lnq_idxes.Add(app);

                                // i
                                Variable for_var1 = (ref1).VarRef;

                                // p
                                Variable linq_var1 = (ref2).VarRef;

                                lnq0 = linq_var1.ParentVar as LINQ;

                                string name = for_var1.Name + "" + linq_var1.Name;
                                Apply for_var2_domain = new Apply(new Reference(DomainFnc), new Term[] { new Reference(used_ref.VarRef), new Number(dim) });

                                // ip
                                Variable for_var2 = new Variable(name, for_var1.TypeVar, for_var2_domain);

                                x_idxes[dim] = new Reference(for_var2);

                                Apply start = Add(Add(for_var2, MaxRange(for_var1.Domain).Minus()), One());
                                Reference end = new Reference(for_var2);
                                Apply linq_var2_domain = Intersect(linq_var1.Domain, Range(start, end));

                                // p
                                Variable linq_var2 = new Variable(linq_var1.Name, linq_var1.TypeVar, linq_var2_domain);
                                var_tbl.Add(linq_var1, linq_var2);

                                lnq_vars.Add(linq_var2);

                                Debug.Assert(u_idxes[dim] is Reference);

                                // ip - p
                                Apply sub2 = Sub(for_var2, linq_var2);

                                // 右辺の dim に ip - p を代入する。
                                subst_tbl.Add(u_idxes[dim] as Reference, sub2);

                                u_idxes[dim] = sub2;
                            }
                            else {
                                // 同じ形の添え字が処理済みの場合

                                x_idxes[dim] = used_ref.Indexes[dim].Clone();
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

                Debug.Assert(x_idxes[dim] != null);
            }

            Term u_right;
            Term right_simple;
            Reference norm_left;

            pr.NormRef = new Reference(fld.Name, fld, x_idxes);

            if (lnq0 != null) {
                // LINQ の添え字がある場合

                asn.Right = NaviRep(asn.Right,
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
                            return true;
                        }

                        return false;
                    }) as Term;

                if (subst_tbl.Keys.Any()) {
                    // 変数の置換がある場合

                    // 右辺の i に ip - p を代入する。
                    u_right = Subst(asn.Right.Clone(), subst_tbl);
                }
                else {
                    // 変数の置換がない場合

                    u_right = asn.Right.Clone();
                }
            }
            else {
                // LINQ の添え字がない場合

                u_right = asn.Right.Clone();
            }

            if ((from e in pr.IndexTypes where e == IndexType.PrevT select e).Any()) {
                // t - 1の添え字がある場合

                // 左辺の t に t+1を代入する。
                norm_left = Tplus1(asn.Left, t_var, null) as Reference;

                // 右辺の t に t+1を代入する。
                u_right = Tplus1(u_right, t_var, null);
            }
            else {
                // t - 1の添え字がない場合

                norm_left = new Reference(asn.Left.Name, asn.Left.VarRef, u_idxes);
            }

            // 右辺の簡約化
            right_simple = SimplifyExpression(u_right.Clone());

            // δE/δu
            pr.DiffE = Diff(new Reference(EFnc), norm_left.Clone());

            // δE/δu * δu/δx
            pr.Delta = MakeLinqMulDiff(pr, lnq_vars, Diff(norm_left.Clone(), pr.NormRef.Clone()));

            // Σ δE/δu * δ(置換右辺)/δx
            pr.DeltaSubst = MakeLinqMulDiff(pr, lnq_vars, Diff(u_right.Clone(), pr.NormRef.Clone()));

            // Σ δE/δu * δ(簡約置換右辺)/δx
            pr.DeltaSimple = MakeLinqMulDiff(pr, lnq_vars, Diff(right_simple.Clone(), pr.NormRef.Clone()));

            Term diff = SetParent(Differential(right_simple, pr.NormRef, null));

            // Σ δE/δu * 微分簡約置換右辺
            pr.RightDiff = MakeLinqMulDiff(pr, lnq_vars, diff);

            // Σ δE/δu * 簡約微分簡約置換右辺
            pr.RightDiffSimple = MakeLinqMulDiff(pr, lnq_vars, SimplifyExpression(diff.Clone()));

            // 伝播の情報を返す。
            return pr;
        }

        void WriteMathJax(StringWriter sw, string file_name) {
/*

*/

            string head = @"<!DOCTYPE html>

<html lang=""en"" xmlns=""http://www.w3.org/1999/xhtml"">
<head>
    <meta charset=""utf-8"" />
    <title>" + file_name + @"</title>
    <script type=""text/x-mathjax-config"">
      MathJax.Hub.Config({
        extensions: [""tex2jax.js""],
        jax: [""input/TeX"",""output/HTML-CSS""],
        tex2jax: {inlineMath: [[""$"",""$""],[""\\("",""\\)""]]}
      });
    </script>

    <script type=""text/javascript"" src=""http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML""></script>
</head>
<body>
";
            string html_dir = HomeDir + "\\html";
            if (! Directory.Exists(html_dir)) {

                Directory.CreateDirectory(html_dir);
            }
            File.WriteAllText(html_dir + "\\" + file_name + ".html", head + sw.ToString() + "\r\n</body></html>", Encoding.UTF8);
        }

        void WriteClassCode(Class cls) {
            Navi(cls,
                delegate (object obj) {
                    if (obj is Reference) {
                        Debug.Assert((obj as Reference).VarRef != null);
                    }
                    else if (obj is Variable) {
                        Debug.Assert((obj as Variable).TypeVar != null);
                    }
                });

            MakeCode mc = new MakeCode(this);
            string header, body;

            mc.ClassCode(cls, out header, out body);

            string html_dir = HomeDir + "\\src\\C";
            if (!Directory.Exists(html_dir)) {

                Directory.CreateDirectory(html_dir);
            }

            File.WriteAllText(html_dir + "\\" + cls.Name + ".h"  , header, Encoding.UTF8);
            File.WriteAllText(html_dir + "\\" + cls.Name + ".cpp", body, Encoding.UTF8);

        }

        /*
            t - 1 => t
            t => t + 1
        */
        Term Tplus1(Term t1, Variable t_var, Dictionary<Variable, Variable> var_tbl) {
            if (var_tbl == null) {
                var_tbl = new Dictionary<Variable, Variable>();
            }

            if (t1 is Reference) {
                Reference r1 = t1 as Reference;
                if(r1.VarRef == t_var) {

                    Debug.Assert(r1.Indexes == null);

                    return new Apply(new Reference(AddFnc), new Term[] { r1.Clone(var_tbl), new Number(1) });
                }
                if (r1.Indexes == null) {
                    return r1.Clone(var_tbl);
                }
                Term[] idx = (from t in r1.Indexes select Tplus1(t, t_var, var_tbl)).ToArray();

                Variable v1;
                if (!var_tbl.TryGetValue(r1.VarRef, out v1)) {
                    v1 = r1.VarRef;
                }

                return new Reference(r1.Name, v1, idx);
            }
            else if (t1 is Number) {
                return t1.Clone(var_tbl);
            }
            else if (t1 is Apply) {
                Apply app = t1 as Apply;
                Term[] args = (from t in app.Args select Tplus1(t, t_var, var_tbl)).ToArray();
                return new Apply(Tplus1(app.Function, t_var, var_tbl) as Reference, args);
            }
            else if (t1 is LINQ) {
                LINQ lnq = t1 as LINQ;
                Variable[] vars = (from v in lnq.Variables select Tplus1Var(v, t_var, var_tbl)).ToArray();
                return new LINQ(vars, Tplus1(lnq.Select, t_var, var_tbl), (lnq.Aggregate == null ? null : Tplus1(lnq.Aggregate, t_var, var_tbl) as Reference));
            }
            else {
                Debug.Assert(false);
            }

            return null;
        }

        Variable Tplus1Var(Variable v, Variable t_var, Dictionary<Variable, Variable> var_tbl) {
            Term domain = (v.Domain == null ? null : Tplus1(v.Domain, t_var, var_tbl));
            Variable v1 = new Variable(v.Name, v.TypeVar, domain);
            var_tbl.Add(v, v1);

            return v1;
        }

        // u[iu, ju, k] = (from p in Range(H) from q in Range(H) select x[iu + p, ju + q] * h[p, q, k]).Sum() + b[k];
        // ix = iu + p   : 0 <= iu <= IU - 1   0 <= p <= H - 1
        //   iu = ix - p  : 0 <= ix - p <= IU - 1    ix - IU + 1 <= p <= ix  max(0, ix - IU + 1) <= p <= min(H - 1, ix)
        // jx = ju + q   : 0 <= ju <= JU - 1   0 <= q <= H - 1
        //   ju = jx - q  : 0 <= jx - q <= JU - 1    jx - JU + 1 <= q <= jx  max(0, jx - JU + 1) <= q <= min(H - 1, jx)
        // H

        Apply Range(Term start, Term end) {
            return new Apply(RangeFnc, start, end);
        }

        Apply Intersect(Term t1, Term t2) {
            if(t1 is Apply && t2 is Apply) {

                Apply app1 = t1 as Apply;
                Apply app2 = t2 as Apply;

                if(app1.Function.VarRef == RangeFnc && app2.Function.VarRef == RangeFnc) {

                    Apply min = new Apply(maxFnc, new Term[] { MinRange(t1), MinRange(t2) });
                    Apply max = new Apply(minFnc, new Term[] { MaxRange(t1), MaxRange(t2) });

                    return Range(min, max);
                }
            }
            return null;
        }

        Term MinRange(Term rng) {
            if (rng is Apply) {
                Apply app = rng as Apply;

                if (app.Function.VarRef == RangeFnc) {

                    if (app.Args.Length == 1) {

                        return Zero();
                    }
                    else {

                        return app.Args[0];
                    }
                }
            }

            throw new Exception();
        }

        Term MaxRange(Term rng) {
            if(rng is Apply) {
                Apply app = rng as Apply;

                if(app.Function.VarRef == RangeFnc) {

                    if(app.Args.Length == 1) {

                        return app.Args[0];
                    }
                    else {

                        return app.Args[1];
                    }
                }
            }

            throw new Exception();
        }

        Apply Diff(Term t, Reference r) {
            return new Apply(DiffFnc, new Term[] { t, r });
        }

        Term[] VariableToReference(object[] args) {
            return (from x in args select (Term)(x is Variable ? new Reference(x as Variable) : x as Term)).ToArray();
        }

        Apply Mul(params object[] args) {
            return new Apply(new Reference(MulFnc), VariableToReference(args));
        }

        Apply Add(params object[] args) {
            return new Apply(new Reference(AddFnc), VariableToReference(args));
        }

        Apply Sub(params object[] args) {
            Term[] v = VariableToReference(args);
            for(int i = 1; i < v.Length; i++) {
                v[i].Value *= -1;
            }

            return new Apply(new Reference(AddFnc), v);
        }

        public static int[] Range(int n) {
            int[] v = new int[n];

            for (int i = 0; i < n; i++) {
                v[i] = i;
            }

            return v;
        }

        /*
            微分
        */
        Term DifferentialLINQ(LINQ lnq, Reference r1, Dictionary<Variable, Variable> var_tbl) {
            Debug.Assert(lnq.Aggregate != null);

            Dictionary<Reference, Dictionary<Reference, Term>> rs = new Dictionary<Reference, Dictionary<Reference, Term>>();
            bool exact = false;
            Navi(lnq.Select,
                delegate (object obj) {
                    if (obj is Reference) {
                        Reference r2 = obj as Reference;

                        if (!(from r in rs.Keys where r.Eq(r2) select r).Any()) {
                                // 処理済みでない場合

                                if (r1.VarRef == r2.VarRef) {
                                    // 同じ変数を参照する場合

                                    if (r1.Eq(obj)) {
                                        // 一致する場合

                                        exact = true;
                                }
                                else {
                                        // 一致しない添え字がある場合

                                    Dictionary<Reference, Term> pairs = new Dictionary<Reference, Term>();
                                    bool ok = true;
                                    for (int i = 0; i < r1.Indexes.Length; i++) {
                                        if (!r1.Indexes[i].Eq(r2.Indexes[i])) {
                                                // 添え字が一致しない場合

                                                if (!(r2.Indexes[i] is Reference)) {
                                                    // 代入候補の変数参照の添え字が変数参照でない場合

                                                    ok = false;
                                                break;
                                            }
                                            else {
                                                    // 両方の添え字が変数参照の場合

                                                    Reference r3 = r2.Indexes[i] as Reference;
                                                IEnumerable<Variable> linq_eq_vars = from v in lnq.Variables where v == r3.VarRef select v;
                                                if (linq_eq_vars.Any()) {
                                                        // LINQの変数の場合

                                                    Variable v = linq_eq_vars.First();
                                                    Debug.Assert(! (from r in pairs.Keys where r.VarRef == v select r).Any());
                                                    pairs.Add(new Reference( v ), r1.Indexes[i]);
                                                }
                                                else {
                                                        // LINQの変数でない場合

                                                        ok = false;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if (ok) {
                                        rs.Add(r2, pairs);
                                    }
                                }
                            }
                        }
                    }
                });

            Debug.Assert(!(exact && rs.Any()), "完全一致の変数参照と代入で一致の変数参照の両方がある場合は未対応");
            if (!(exact || rs.Any())) {
                // 完全一致や代入で一致の変数参照がない場合

                return Zero();
            }

            // LINQをコピーする。
            LINQ lnq1 = lnq.Clone(var_tbl);

            if (exact) {
                // 完全一致の変数参照がある場合

                // select句を微分する。
                Term dif1 = Differential(lnq1.Select, r1, var_tbl);

                if (lnq.Aggregate.Name == "Sum") {
                    return lnq1;
                }
                else if (lnq.Aggregate.Name == "Prod") {
                    Debug.Write("の微分は未実装");
                    return lnq1;
                }
                if (lnq.Aggregate.Name == "Max") {
                    Debug.Write("の微分は未実装");
                    return lnq1;
                }
                else {
                    Debug.Assert(false);
                }
            }
            else {
                // 代入で一致の変数参照がある場合

                Debug.Assert(rs.Keys.Count == 1, "代入で一致の変数参照は1種類のみ実装");
                Dictionary<Reference, Term> subst_tbl = rs.First().Value;
                Debug.Assert(subst_tbl.Count == lnq.Variables.Length, "LINQの全変数に代入する。");

                // LINQのselect句の変数参照に代入する。
                Term subst_sel = Subst(lnq1.Select, subst_tbl, var_tbl);

                // LINQの変数に代入をしたselect句を微分する。
                Term dif1 = Differential(subst_sel, r1, var_tbl);

                if (lnq.Aggregate.Name == "Sum") {
                    return dif1;
                }
                else if (lnq.Aggregate.Name == "Prod") {
                    Debug.Write("の微分は未実装");
                    return dif1;
                }
                if (lnq.Aggregate.Name == "Max") {
                    Debug.Write("の微分は未実装");
                    return lnq1;
                }
                else {
                    Debug.Assert(false);
                }
            }

            return null;
        }

        /*
            微分
        */
        Term Differential(Term t1, Reference r1, Dictionary<Variable, Variable> var_tbl_up) {
            if (t1 is Reference) {
                // 変数参照の場合

                if (t1.Eq(r1)) {
                    return One();
                }
                else {
                    return Zero();
                }
            }
            else if (t1 is Number) {
                // 数値の場合

                return Zero();
            }

            Dictionary<Variable, Variable> var_tbl = (var_tbl_up == null ? new Dictionary<Variable, Variable>() : new Dictionary<Variable, Variable>(var_tbl_up));
            if (t1 is Apply) {
                // 関数適用の場合

                Apply app = t1 as Apply;

                Term[] diffs = (from t in app.Args select Differential(t, r1, var_tbl)).ToArray();

                if(app.Function.VarRef == AddFnc) {
                    // 加算の場合

                    return Add(diffs);
                }
                else if (app.Function.VarRef == MulFnc) {
                    // 乗算の場合

                    Term[] args = new Term[app.Args.Length];
                    foreach(int i in Range(app.Args.Length)) {
                        args[i] = Mul((from j in Range(app.Args.Length) select (i == j ? diffs[i] : app.Args[j].Clone(var_tbl))).ToArray());
                    }

                    return Add(args);
                }
                else if (app.Function.VarRef == MaxPoolFnc) {
                    return new Apply(MaxPoolPrimeFnc, app.Args[0].Clone(var_tbl));
                }
                else if (app.Function.Name == "σ") {

                    Term[] args = (from t in app.Args select t.Clone(var_tbl)).ToArray();
                    return Mul(new Apply(new Reference(σ_prime), args), diffs[0]);
                }
                else if (app.Function.Name == "tanh") {

                    Term[] args = (from t in app.Args select t.Clone(var_tbl)).ToArray();
                    return Mul(new Apply(new Reference(tanh_prime), args), diffs[0]);
                }
                else if (app.Function.Name == "Mat" || app.Function.Name == "Row" || app.Function.Name == "C") {
                    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 未実装 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    return Zero();
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (t1 is LINQ) {
                return DifferentialLINQ(t1 as LINQ, r1, var_tbl);
            }
            else {
                Debug.Assert(false);
            }

            Debug.Assert(false);
            return null;
        }

        Term Subst(Term t1, Dictionary<Reference, Term> subst_tbl, Dictionary<Variable, Variable> var_tbl = null) {
            return NaviRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;
                    if (obj is Reference) {
                        Reference r2 = obj as Reference;

                        Term t2;
                        var vref = from r in subst_tbl.Keys where r.Eq(r2) select r;
                        if (vref.Any()) {
                            t2 = subst_tbl[vref.First()];
                            Term t3 = t2.Clone(var_tbl);
                            t3.Parent = r2.Parent;
                            ret = t3;
                            return true;
                        }
                    }

                    return false;
                }) as Term;
        }

        Term SetParent(Term t1) {
            return NaviRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;

                    return false;
                }) as Term;

        }

        /*
            数式の簡約化
        */
        Term SimplifyExpression(Term t1) {
            return NaviRep(t1,
                delegate (object obj, out object ret) {
                    ret = obj;

                    if (obj is Apply) {
                        // 関数適用の場合

                        Apply app = obj as Apply;

                        // 引数を簡約化する。
                        Term[] args1 = (from t in app.Args select SimplifyExpression(t)).ToArray();


                        if (app.IsAdd() || app.IsMul()) {

                            List<Term> args2 = new List<Term>();

                            foreach(Term t2 in args1) {
                                if (t2 is Apply && (t2 as Apply).Function.VarRef == app.Function.VarRef) {
                                    // 引数が同じ演算の場合

                                    args2.AddRange(t2.ToApply().Args);
                                }
                                else {
                                    // 引数が加算や減算でない場合

                                    args2.Add(t2);
                                }
                            }

                            for(int i = 0; i < args2.Count; i++) {

                                for (int j = i + 1; j < args2.Count;) {
                                    if(args2[i].EqBody(args2[j])) {
                                        // 本体が同じ項がある場合

                                        if (app.IsAdd()) {
                                            // 加算の場合

                                            // 係数を加算する。
                                            args2[i].Value += args2[j].Value;
                                        }
                                        else {
                                            // 乗算の場合

                                            // 係数を乗算する。
                                            args2[i].Value *= args2[j].Value;
                                        }

                                        // 同じ項を取り除く
                                        args2.RemoveAt(j);
                                    }
                                    else {
                                        // 同じ項がない場合

                                        j++;
                                    }
                                }
                            }

                            if (app.IsAdd()) {
                                // 加算の場合

                                // 係数が0の項を除く。
                                Term[] args3 = (from t in args2 where t.Value != 0 select t).ToArray();

                                switch (args3.Length) {
                                case 0:
                                    ret = Zero();
                                    return true;

                                case 1:
                                    ret = args3[0];

                                    return true;

                                default:
                                    ret = Add(args3);
                                    return true;
                                }
                            }
                            else {
                                // 乗算の場合

                                // 引数の係数をすべてかけてまとめる。
                                double n = (from t in args2 select t.Value).Aggregate((x, y) => x * y);
                                foreach(Term t in args2) {
                                    t.Value = 1;
                                }

                                if (n == 0) {
                                    // 係数の積が0の場合

                                    // 結果は0
                                    ret = Zero();
                                    return true;
                                }

                                // 定数を除く。
                                Term[] args3 = (from t in args2 where ! (t is Number) select t).ToArray();

                                switch (args3.Length) {
                                case 0:
                                    Debug.Assert(args2.Count == 1 && args2[0] is Number && (args2[0] as Number).Value == 1);

                                    ret = One();
                                    break;

                                case 1:
                                    ret = args3[0];
                                    break;

                                default:
                                    ret = Mul(args3);
                                    break;
                                }

                                (ret as Term).Value = app.Value * n;
                                return true;
                            }
                        }

                        ret = new Apply(app.Function.VarRef, args1);

                        return true;
                    }

                    return false;
                }) as Term;
        }

        /*
            誤差逆伝播法
        */
        Term Backpropagation(Term t1, Variable v) {
            return null;
        }

        string MathJax(Term t1) {
            if (!(t1 is Number)) {

                if (t1.Value == 1) {

                    return MathJaxBody(t1);
                }
                else if (t1.Value == -1) {

                    return "- " + MathJaxBody(t1);
                }
                else {

                    return t1.Value.ToString() + @" \cdot " + MathJaxBody(t1);
                }
            }
            else {

                return MathJaxBody(t1);
            }
        }

        string MathJaxBody(Term t1) {
            if (t1 is Reference) {
                Reference r1 = t1 as Reference;
                if(r1.Indexes == null) {
                    return r1.Name;
                }
                else {

                    switch (r1.Indexes.Length) {
                    case 1:
                        return r1.Name + "_{" + MathJax(r1.Indexes[0]) + "}";
                    case 2:
                        return r1.Name + "_{" + MathJax(r1.Indexes[0]) + "}^{" + MathJax(r1.Indexes[1]) + "}";
                    case 3:
                        return r1.Name + "_{" + MathJax(r1.Indexes[0]) + "}^{" + MathJax(r1.Indexes[1]) + ", " + MathJax(r1.Indexes[2]) + "}";
                    }
                    return r1.Name + "[" + string.Join(", ", from x in r1.Indexes select MathJax(x)) + "]";
                }
            }
            else if (t1 is Number) {
                Number n = t1 as Number;

                return n.Value.ToString();
            }
            else if (t1 is Apply) {
                Apply app = t1 as Apply;

                if ("+-*/%".Contains(app.Function.Name[0])) {
                    string s;

                    Debug.Assert(app.Args.Length != 1);
                    if (app.IsMul()) {

                        s = string.Join(@" \cdot ", from x in app.Args select MathJax(x));
                    }
                    else if (app.IsAdd()) {

                        s = string.Join(" ", from x in app.Args select (x == app.Args[0] || x.Value < 0 ? "" : "+ ") + MathJax(x));
                    }
                    else {

                        s = string.Join(" " + app.Function.Name + " ", from x in app.Args select MathJax(x));
                    }

                    if (app.Parent is Apply && (app.Parent as Apply).Precedence() <= app.Precedence()) {
                        return "(" + s + ")";
                    }
                    else {
                        return s;
                    }
                }
                else {
                    string name = app.Function.Name;

                    if(app.Function.VarRef == DiffFnc) {

                        if (MathJaxDelta && app.Args[0] is Reference && (app.Args[0] as Reference).VarRef == EFnc) {

                            return string.Format(@"\delta^{{ {0} }}", MathJax(app.Args[1]));
                        }
                        string arg0 = MathJax(app.Args[0]);
                        if (app.Args[0] is Apply && ! Char.IsLetter( (app.Args[0] as Apply).Function.Name[0]) ) {

                            arg0 = "(" + arg0 + ")";
                        }
                        return string.Format(@"\frac{{ \partial {0} }}{{ \partial {1} }}", arg0, MathJax(app.Args[1]));
                    }
                    if(app.Function.VarRef == σ_prime) {
                        name = "σ'";
                    }
                    if (app.Function.VarRef == tanh_prime) {
                        name = "tanh'";
                    }
                    return name + "(" + string.Join(", ", from x in app.Args select MathJax(x)) + ")";
                }
            }
            else if (t1 is LINQ) {
                
                LINQ lnq = t1 as LINQ;

                string s = "";

                var vv = from v in lnq.Variables where v.Domain is Apply && (v.Domain as Apply).Function.Name == "Range" select (v.Domain as Apply).Args[0];
                Debug.Assert(vv.Count() == lnq.Variables.Count());
                Dictionary<Variable, Term> doms = lnq.Variables.ToDictionary(v => v, v => (v.Domain as Apply).Args[0]);

                if (lnq.Aggregate.Name == "Sum") {

                    foreach (Variable v in lnq.Variables) {
                        s += @"\displaystyle \sum_{" + v.Name + " }^{ " + MathJax(doms[v]) + " } ";
                    }
                }
                else if (lnq.Aggregate.Name == "Prod") {

                    foreach (Variable v in lnq.Variables) {
                        s += "\\prod_{" + v.Name + " }^{ " + MathJax(doms[v]) + " } ";
                    }
                }
                else if (lnq.Aggregate.Name == "Max") {

                    foreach (Variable v in lnq.Variables) {
                        s += @"\mathop{max}_{ \small " + v.Name + @" }^{ \small " + MathJax(doms[v]) + " } ";
                    }
                }
                else {
                    Debug.Assert(false);
                }
                s += MathJax(lnq.Select);

                return s;              
            }
            else {
                Debug.Assert(false);
            }

            return null;
        }

        Term[] AllTerms(object root) {
            List<Term> v = new List<Term>();
            Navi(root,
                delegate (object obj) {
                    if (obj is Term) {
                        v.Add(obj as Term);
                    }
                });
            return v.ToArray();
        }

        Reference[] Refs(object root) {
            List<Reference> v = new List<Reference>();
            Navi(root,
                delegate (object obj) {
                    if (obj is Reference) {
                        v.Add(obj as Reference);
                    }
                });
            return v.ToArray();
        }

        Statement ParentStatement(Term t1) {
            for (Object obj = t1.Parent; ; ) {
                Debug.Assert(obj != null);

                if (obj is Statement) {
                    return obj as Statement;
                }
                else if (obj is Term) {
                    obj = (obj as Term).Parent;
                }
                else if (obj is Variable) {
                    obj = (obj as Variable).ParentVar;
                }
                else {
                    Debug.Assert(false);
                }
            }
        }

        ForEach[] AncestorForEach(Object o) {
            Statement stmt;

            if(o is Statement) {
                stmt = o as Statement;
            }
            else {
                Debug.Assert(o is Term);
                stmt = ParentStatement(o as Term);
            }

            List<ForEach> vfor = new List<ForEach>();
            Debug.Assert(stmt.ParentStmt is ForEach);

            ForEach for1 = stmt.ParentStmt as ForEach;
            vfor.Add(for1);
            while (for1.ParentStmt is ForEach) {
                for1 = for1.ParentStmt as ForEach;
                vfor.Add(for1);
            }
            Debug.Assert(for1.ParentStmt is BlockStatement);

            vfor.Reverse();
            return vfor.ToArray();
        }

        List<object> Ancestors(List<Term> terms) {
            HashSet<Term> pending = new HashSet<Term>(terms);
            HashSet<object> processed = new HashSet<object>();

            while (pending.Any()) {
                HashSet<object> tmp = new HashSet<object>(from x in pending where x.Parent != null select x.Parent);

                foreach(object t in pending) {
                    processed.Add(t);
                }

                pending = new HashSet<Term>(from t in tmp where t is Term && ! processed.Contains(t) select (Term)t);
            }

            return new List<object>(processed);
        }

        public static void Navi(object obj, NaviAction before, NaviAction after = null) {
            if (obj == null) {
                return;
            }

            if(before != null) {
                before(obj);
            }

            if (obj is Term) {

                if (obj is Reference) {
                    Reference r1 = obj as Reference;

                    if(r1.Indexes != null) {
                        foreach(Term t in r1.Indexes) {
                            Navi(t, before, after);
                        }
                    }
                }
                else if (obj is Number) {
                }
                else if (obj is Apply) {
                    Apply app = obj as Apply;
                    Navi(app.Function, before, after);
                    foreach (Term t in app.Args) {
                        Navi(t, before, after);
                    }
                }
                else if (obj is LINQ) {
                    LINQ lnq = obj as LINQ;
                    foreach (Variable v in lnq.Variables) {
                        Navi(v, before, after);
                    }
                    Navi(lnq.Select, before, after);
                    Navi(lnq.Aggregate, before, after);
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Variable) {
                Variable v = obj as Variable;

                Navi(v.Domain, before, after);

                if(obj is Function) {
                    Function fnc = obj as Function;
                    foreach(Variable p in fnc.Params) {
                        Navi(p, before, after);
                    }
                    Navi(fnc.BodyStatement, before, after);
                }
            }
            else if (obj is Statement) {
                if (obj is Assignment) {
                    Assignment asn = obj as Assignment;
                    Navi(asn.Left, before, after);
                    Navi(asn.Right, before, after);
                }
                else if(obj is Return) {
                    Return ret = obj as Return;
                    Navi(ret.Value, before, after);
                }
                else if (obj is ForEach) {
                    ForEach for1 = obj as ForEach;
                    Navi(for1.LoopVariable, before, after);
                    foreach (Statement s in for1.Statements) {
                        Navi(s, before, after);
                    }
                }
                else if (obj is BlockStatement) {
                    BlockStatement blc1 = obj as BlockStatement;
                    foreach (Statement s in blc1.Statements) {
                        Navi(s, before, after);
                    }
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Class) {
                Class cls = obj as Class;
                foreach(Variable fld in cls.Fields) {
                    Navi(fld, before, after);
                }
                foreach (Function fnc in cls.Functions) {
                    Navi(fnc, before, after);
                }
            }
            else {
                Debug.Assert(false);
            }

            if(after != null) {

                after(obj);
            }
        }

        object NaviRep(object obj, NaviFnc before, NaviFnc after = null) {
            if (obj == null) {
                return null;
            }

            object ret = obj;

            if (before != null) {

                bool done = before(obj, out ret);
                if (done) {

                    return ret;
                }
            }

            if (obj is Term) {

                if (obj is Reference) {
                    Reference r1 = obj as Reference;

                    if (r1.Indexes != null) {
                        r1.Indexes = (from t in r1.Indexes select NaviRep(t, before, after) as Term).ToArray();
                        
                        foreach(Term t in r1.Indexes) {
                            t.Parent = obj;
                        }
                    }
                }
                else if (obj is Number) {
                }
                else if (obj is Apply) {
                    Apply app = obj as Apply;
                    app.Function = NaviRep(app.Function, before, after) as Reference;
                    app.Args = (from t in app.Args select NaviRep(t, before, after) as Term).ToArray();

                    app.Function.Parent = app;
                    foreach (Term t in app.Args) {
                        t.Parent = obj;
                    }
                }
                else if (obj is LINQ) {
                    LINQ lnq = obj as LINQ;
                    lnq.Variables = (from v in lnq.Variables select NaviRep(v, before, after) as Variable).ToArray();
                    lnq.Select = NaviRep(lnq.Select, before, after) as Term;
                    lnq.Aggregate = NaviRep(lnq.Aggregate, before, after) as Reference;

                    foreach(Variable v in lnq.Variables) {
                        v.ParentVar = obj;
                    }
                    lnq.Select.Parent = obj;
                    if (lnq.Aggregate != null) {
                        lnq.Aggregate.Parent = obj;
                    }
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Variable) {
                Variable v = obj as Variable;

                v.Domain = NaviRep(v.Domain, before, after) as Term;
                if(v.Domain != null) {
                    v.Domain.Parent = obj;
                }

                if (obj is Function) {
                    Function fnc = obj as Function;
                    fnc.Params = (from p in fnc.Params select NaviRep(p, before, after) as Variable).ToList();
                    fnc.BodyStatement = NaviRep(fnc.BodyStatement, before, after) as BlockStatement;

                    foreach (Variable p in fnc.Params) {
                        p.ParentVar = fnc;
                    }
                    fnc.BodyStatement.ParentStmt = fnc;
                }
            }
            else if (obj is Statement) {
                if (obj is Assignment) {
                    Assignment asn = obj as Assignment;
                    asn.Left = NaviRep(asn.Left, before, after) as Reference;
                    asn.Right = NaviRep(asn.Right, before, after) as Term;

                    asn.Left.Parent = obj;
                    asn.Right.Parent = obj;
                }
                else if (obj is Return) {
                    Return ret_stmt = obj as Return;
                    ret_stmt.Value = NaviRep(ret_stmt.Value, before, after) as Term;

                    if(ret_stmt.Value != null) {

                        ret_stmt.Value.Parent = obj;
                    }
                }
                else if (obj is ForEach) {
                    ForEach for1 = obj as ForEach;
                    for1.LoopVariable = NaviRep(for1.LoopVariable, before, after) as Variable;
                    for1.Statements = (from s in for1.Statements select NaviRep(s, before, after) as Statement).ToList();

                    for1.LoopVariable.ParentVar = obj;
                    foreach(Statement stmt in for1.Statements) {
                        stmt.ParentStmt = obj;
                    }
                }
                else if (obj is BlockStatement) {
                    BlockStatement blc1 = obj as BlockStatement;
                    blc1.Statements = (from s in blc1.Statements select NaviRep(s, before, after) as Statement).ToList();
                    foreach (Statement stmt in blc1.Statements) {
                        stmt.ParentStmt = obj;
                    }
                }
                else {
                    Debug.Assert(false);
                }
            }
            else if (obj is Class) {
                Class cls = obj as Class;
                cls.Fields = (from fld in cls.Fields select NaviRep(fld, before, after) as Variable).ToList();
                cls.Functions = (from fnc in cls.Functions select NaviRep(fnc, before, after) as Function).ToList();

                foreach (Variable fld in cls.Fields) {
                    fld.ParentVar = cls;
                }
                foreach (Function fnc in cls.Functions) {
                    fnc.ParentVar = cls;
                }
            }
            else {
                Debug.Assert(false);
            }

            if (after != null) {

                after(obj, out ret);
            }

            return ret;
        }      
    }

    public class SubstPair {
        public Variable VarPair;
        public Term TermPair;

        public SubstPair(Variable var, Term term) {
            VarPair = var;
            TermPair = term;
        }
    }

}