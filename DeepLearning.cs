using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace mkfn {
    public delegate void NaviAction(object self);
    public delegate bool NaviFnc(object self, out object ret);

    //------------------------------------------------------------ TProject
    public partial class mkfn {
        Number Zero = new Number(0);
        Number One = new Number(1);
        public Variable AddFnc = new Variable("+", null, null);
        public Variable SubFnc = new Variable("-", null, null);
        public Variable MulFnc = new Variable("*", null, null);
        Variable DivFnc = new Variable("/", null, null);
        Variable SumFnc = new Variable("Sum", null, null);
        Variable ProdFnc = new Variable("Prod", null, null);
        Variable MaxFnc = new Variable("Max", null, null);
        Variable maxFnc = new Variable("max", null, null);
        Variable minFnc = new Variable("min", null, null);
        Variable NewFnc = new Variable("new", null, null);
        Variable DomainFnc = new Variable("Domain", null, null);
        Variable RangeFnc;
        Variable DiffFnc = new Variable("Diff", null, null);
        Variable EFnc = new Variable("E", null, null);
        Variable σ_prime;
        Variable tanh_prime;

        public void DeepLearning() {

            var intArray = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            Debug.WriteLine( string.Join(", ", (from x in intArray select (x*10).ToString()) ));

            Debug.WriteLine("深層学習");

            Class layer = (from cls in AppClasses where cls.Name == "Layer" select cls).First();

            RangeFnc = (from f in layer.Functions where f.Name == "Range" select f).First();

            Class[] layers = (from cls in AppClasses where cls.IsSubClass(layer) select cls).ToArray();

            StringWriter sw = new StringWriter();

            σ_prime = (from fnc in layer.Functions where fnc.Name == "σ_prime" select fnc).First();
            tanh_prime = (from fnc in layer.Functions where fnc.Name == "tanh_prime" select fnc).First();

            // アプリのクラスの親クラスに対し
            foreach (Class cls in layers) {
                //if (cls.Name != "ConvolutionalLayer" && cls.Name != "MaxPoolingLayer") continue;//????????????????????????

                Debug.WriteLine("layer : {0}", cls.Name, "");

                Function fnc = (from f in cls.Functions where f.Name == "Forward" select f).First();

                Variable x_var = (from f in cls.Fields where f.Name == "x" select f).First();
                Variable y_var = (from f in cls.Fields where f.Name == "y" select f).First();
                Debug.Assert(x_var.TypeVar is ArrayType && y_var.TypeVar is ArrayType);

                Debug.Assert(fnc.Statement.Statements.Count == 1 && fnc.Statement.Statements[0] is ForEach);

                ForEach top_for = (ForEach)fnc.Statement.Statements[0];

                Variable t_var = null;
                if(top_for.LoopVariable.Name == "t") {
                    t_var = top_for.LoopVariable;
                }

                List<Term> all_terms = new List<Term>();
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

                Reference[] all_refs = (from t in all_terms where t is Reference select t as Reference).ToArray();

                //------------------------------------------------------------ 順伝播
                sw.WriteLine("<hr/>");
                sw.WriteLine("<h4 style='color : red;'>{0}順伝播</h4>", cls.Name, "");
                sw.WriteLine("$$");
                sw.WriteLine(string.Join("\r\n \\\\ \r\n", from asn in all_asns select MathJax(asn.Left) + " = " + MathJax(asn.Right)));
                sw.WriteLine("$$");


                List<Reference> lefts = new List<Reference>();
//???????????                lefts.Add(x_var);

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
                    lefts.Add(left);
                }

                Apply t_sub_1 = null;
                if(t_var != null) {
                    t_sub_1 = Sub(new Term[] { new Reference(t_var), One });
                }

                // すべてのフィールドに対し
                lefts.Clear();
                foreach (Variable fld in cls.Fields) {
                    
                    // フィールドの値を使用する変数参照のリスト
                    Reference[] vref = (from r in all_refs where r.VarRef == fld && r.Indexes != null && r.Used() select r).ToArray();

                    if (vref.Any()) {
                        // フィールドの値を使用する変数参照がある場合

                        Reference left = MakeDifferentialReference(t_var, t_sub_1, fld, vref);
                        lefts.Add(left);
                    }
                }

                // すべての代入文の左辺の変数参照に対し
                foreach (Reference left in lefts) {
                    //Debug.Assert(left.Defined());
                    List<Reference> prop = new List<Reference>();
                    List<Reference> prop_t_1 = new List<Reference>();
                    List<Reference> prop_plus_linq = new List<Reference>();
                    Dictionary<Reference, LINQ> prop_plus_linq_diff = new Dictionary<Reference, LINQ>();
                    Dictionary<Reference, LINQ> prop_plus_linq_diff_right = new Dictionary<Reference, LINQ>();

                    // 変数を使用している変数参照に対し
                    foreach (Reference ref_use in (from r in all_refs where r.VarRef == left.VarRef && r.Used() && r.Indexes != null select r)) {
                        Debug.Assert(ref_use.Used());

                        // 変数を使用している変数参照の親の文
                        Statement stmt = ParentStatement(ref_use);
                        if(stmt is Assignment) {
                            // 変数を使用している変数参照の親の文が代入文の場合

                            // 変数を使用している代入文
                            Assignment asn_use = stmt as Assignment;
                            Debug.Assert(asn_use.Left is Reference);

                            // 変数を使用している代入文の左辺の変数参照
                            Reference left_use = asn_use.Left as Reference;

                            // 変数を使用している代入文の祖先のForEachのリスト
                            ForEach[] vfor = AncestorForEach(asn_use);

                            // 変数参照の次元
                            int dim_cnt = ref_use.Indexes.Length;
                            int ref_linq = 0;
                            int t_1 = 0;
                            ForEach plus_linq_foreach = null;
                            LINQ plus_linq_linq = null;
                            List<int> plus_linq_dim = new List<int>();

                            // 変数参照の各添え字に対し
                            for (int dim = 0; dim < dim_cnt; dim++) {
                                if (ref_use.Indexes[dim] is Reference) {
                                    // 添え字が変数参照の場合

                                    // 変数参照の添え字の変数参照
                                    Reference idx_ref = ref_use.Indexes[dim] as Reference;
                                    if (idx_ref.Indexes == null) {
                                        // 変数参照の添え字が添え字を持たない場合

                                        if (idx_ref.VarRef.ParentVar is LINQ) {
                                            // 変数参照の添え字がLINQのループ変数の場合

                                            // 変数参照の添え字がループ変数のLINQ
                                            LINQ lnq = idx_ref.VarRef.ParentVar as LINQ;
                                            Debug.Assert(lnq.Aggregate != null);
                                            ref_linq++;

                                            if (lnq.Aggregate.Name == "Sum") {
                                                Debug.Write("");
                                            }
                                            else if (lnq.Aggregate.Name == "Prod") {
                                                Debug.Write("");
                                            }
                                            else if (lnq.Aggregate.Name == "Max") {
                                                Debug.Assert(false, "未実装");
                                            }
                                            else {
                                                Debug.Assert(false);
                                            }
                                        }
                                        else if(idx_ref.VarRef.ParentVar is ForEach){
                                            // 変数参照の添え字がForEachのループ変数の場合
                                        }
                                        else {
                                            // 変数参照の添え字がLINQやForEachのループ変数でない場合

                                            Debug.Assert(false);
                                        }
                                    }
                                    else {
                                    }
                                }
                                else {
                                    // 添え字が変数参照でない場合
                                    Debug.Assert(ref_use.Indexes[dim] is Apply);

                                    // 添え字の関数適用
                                    Apply app = ref_use.Indexes[dim] as Apply;
                                    Debug.Assert(app.Args.Length == 2 && app.Args[0] is Reference);

                                    // 添え字の関数適用の最初の引数
                                    Reference arg_ref1 = app.Args[0] as Reference;
                                    Debug.Assert(arg_ref1.VarRef == vfor[dim].LoopVariable);

                                    if (app.Function.Name == "-") {
                                        // 添え字が"-"式の場合

                                        Debug.Assert(app.Args[1] is Number);

                                        // 添え字の関数適用の2番目の引数
                                        Number n = app.Args[1] as Number;
                                        Debug.Assert(arg_ref1.VarRef == t_var && n.Value == 1);
                                        // t - 1
                                        t_1++;
                                    }
                                    else if (app.Function.Name == "+") {
                                        // 添え字が"+"式の場合

                                        Debug.Assert(app.Args[1] is Reference);

                                        // 添え字の関数適用の2番目の引数
                                        Reference arg_ref2 = app.Args[1] as Reference;
                                        Debug.Assert(dim < vfor.Length && arg_ref1.VarRef == vfor[dim].LoopVariable);
                                        Debug.Assert(arg_ref2.VarRef.ParentVar is LINQ);

                                        // 添え字の関数適用の2番目の引数はLINQのループ変数
                                        LINQ lnq = arg_ref2.VarRef.ParentVar as LINQ;

                                        if (lnq.Aggregate.Name == "Sum") {
                                            Debug.Write("");
                                        }
                                        else if (lnq.Aggregate.Name == "Max") {
                                            Debug.Write("");
                                        }
                                        else {
                                            Debug.Assert(false);
                                        }

                                        // i + p, j + q
                                        plus_linq_dim.Add(dim);
                                        if (plus_linq_foreach == null) {

                                            // 添え字の関数適用の最初の引数はforeachのループ変数
                                            plus_linq_foreach = vfor[dim];

                                            // 添え字の関数適用の2番目の引数はLINQのループ変数
                                            plus_linq_linq = lnq;
                                        }
                                        else {

                                            //Debug.Assert(plus_linq_foreach == vfor[dim]);
                                            Debug.Assert(plus_linq_linq == lnq);
                                        }
                                    }
                                    else {
                                        Debug.Assert(false);
                                    }
                                }
                            }
                            if (t_1 != 0 && ref_linq != 0) {

                            }
                            else if (ref_linq != 0 && plus_linq_foreach != null || plus_linq_foreach != null && t_1 != 0) {
                                Debug.Assert(false);
                            }

                            if (t_1 != 0) {

                                if (!(from r in prop_t_1 where r.Eq(left_use) select r).Any()) {
                                    // t-1の伝播先リストにない場合

                                    // t-1の伝播先リストに追加する。
                                    prop_t_1.Add(left_use);
                                }
                            }
                            else if (plus_linq_foreach != null) {

                                if (!(from r in prop_plus_linq where r.Eq(left_use) select r).Any()) {
                                    // i+pの伝播先リストにない場合

                                    // i+pの伝播先リストに追加する。
                                    prop_plus_linq.Add(left_use);

                                    LINQ lnq1, lnq2;
                                    plus_linq(ref_use, plus_linq_dim, out lnq1, out lnq2);
                                    prop_plus_linq_diff.Add(left_use, lnq1);
                                    prop_plus_linq_diff_right.Add(left_use, lnq2);
                                }
                            }
                            else {
                                if (!(from r in prop where r.Eq(left_use) select r).Any()) {
                                    // 伝播先リストにない場合

                                    // 伝播先リストに追加する。
                                    prop.Add(left_use);
                                }
                            }
                        }
                    }

                    if(prop.Any() || prop_t_1.Any() || prop_plus_linq.Any()) {

                        // 右辺
                        Dictionary<Reference, Term> use_right = prop.ToDictionary(r => r, r => (r.Parent as Assignment).Right);

                        // 左辺を+1する。
                        Dictionary<Reference, Reference> use_left_inc = prop_t_1.ToDictionary(r => r, r => Tplus1(r, t_var, null) as Reference);

                        // 右辺を+1する。
                        Dictionary<Reference, Term> use_right_inc = prop_t_1.ToDictionary(r => r, r => Tplus1((r.Parent as Assignment).Right, t_var, null));

                        // 右辺を+1して簡約化する。
                        Dictionary<Reference, Term> use_right_inc_simple = prop_t_1.ToDictionary(r => r, r => SimplifyExpression(use_right_inc[r].Clone(null)));


                        foreach (Reference r in prop_plus_linq) {
                            sw.WriteLine("<hr/>");
                            sw.WriteLine("$$");
                            sw.WriteLine(MathJax(prop_plus_linq_diff[r]));
                            sw.WriteLine(@"\\ =");
                            sw.WriteLine(MathJax(prop_plus_linq_diff_right[r]));
                            sw.WriteLine("$$");
                        }


                        // tとt+1の合併
                        Debug.Assert(! prop.Intersect(prop_t_1).Any());
                        List<Reference> prop_union = prop.Union(prop_t_1).ToList();

                        // tとt+1の左辺の合併
                        Dictionary<Reference, Reference> use_left_union = prop_union.ToDictionary(r => r, r => (prop.Contains(r) ? r : use_left_inc[r]));

                        // tとt+1の右辺の合併
                        Dictionary<Reference, Term> use_right_union = prop_union.ToDictionary(r => r, r => (prop.Contains(r) ? use_right[r] : use_right_inc_simple[r]));

                        // 右辺を微分する。
                        Dictionary<Reference, Term> use_right_diff = prop_union.ToDictionary(r => r, r => SetParent( Differential(use_right_union[r], left, null)));

                        // 右辺を微分して簡約化する。
                        Dictionary<Reference, Term> use_right_diff_simple = prop_union.ToDictionary(r => r, r => SimplifyExpression(use_right_diff[r].Clone(null)));


                        sw.WriteLine("<hr/>");
                        sw.WriteLine("<div style='font-size:120%;'>");                        
                        sw.WriteLine("$$");
                        sw.WriteLine(@"\frac{{ \partial E }}{{ \partial {0} }}", MathJax(left), "");
                        sw.WriteLine("$$");
                        sw.WriteLine("</div>");                        

                        //------------------------------------------------------------ 順伝播先の変数の偏微分から計算式を作る。
                        sw.WriteLine("<h5>順伝播先の変数の偏微分から計算式を作る。</h5>");
                        sw.WriteLine("$$");
                        sw.Write("= ");

                        sw.WriteLine(string.Join(" + ", from r in prop.Union(use_left_inc.Values)
                                                        select string.Format(@"\frac{{ \partial E }}{{ \partial {0} }} \cdot \frac{{ \partial {0} }}{{ \partial {1} }}",
                                                        MathJax(r), MathJax(left))));

                        sw.WriteLine("$$");

                        //------------------------------------------------------------  順伝播先の変数に定義式を代入する。
                        sw.WriteLine("<h5>順伝播先の変数に定義式を代入する。</h5>");
                        sw.WriteLine("$$");
                        sw.Write("= ");

                        sw.WriteLine(string.Join(@" \\ + ", from r in prop select string.Format(@"\delta^{{ {0} }} \cdot \frac{{ \partial ({1}) }}{{ \partial {2} }}",
                            MathJax(r), MathJax(use_right[r]), MathJax(left))));

                        if (prop.Any() && prop_t_1.Any()) {
                            sw.WriteLine(" + ");
                        }

                        sw.WriteLine(string.Join(@" \\ + ", from r in prop_t_1
                                                            select string.Format(@"\delta^{{ {0} }} \cdot \frac{{ \partial ({1}) }}{{ \partial {2} }}",
                                                            MathJax(use_left_inc[r]), MathJax(use_right_inc[r]), MathJax(left))));
                        sw.WriteLine("$$");

                        //------------------------------------------------------------  (t + 1) − 1 を t に簡約化する。
                        sw.WriteLine("<h5>(t + 1) − 1 を t に簡約化する。</h5>");
                        sw.WriteLine("$$");
                        sw.Write("= ");

                        sw.WriteLine(string.Join(@" \\ + ", from r in prop
                                                            select string.Format(@"\delta^{{ {0} }} \cdot \frac{{ \partial ({1}) }}{{ \partial {2} }}",
                                                            MathJax(r), MathJax(use_right[r]), MathJax(left))));

                        if (prop.Any() && prop_t_1.Any()) {
                            sw.WriteLine(" + ");
                        }

                        sw.WriteLine(string.Join(@" \\ + ", from r in prop_t_1
                                                            select string.Format(@"\delta^{{ {0} }} \cdot \frac{{ \partial ({1}) }}{{ \partial {2} }}",
                                                            MathJax(use_left_inc[r]), MathJax(use_right_inc_simple[r]), MathJax(left))));
                        sw.WriteLine("$$");

                        //------------------------------------------------------------  微分の計算をする。
                        sw.WriteLine("<h5>微分の計算をする。</h5>");
                        sw.WriteLine("$$");
                        sw.Write("= ");

                        sw.WriteLine(string.Join(" \\\\ \r\n + ", from r in prop_union
                                                            select string.Format(@"\delta^{{ {0} }} \cdot ( {1} )",
                                                            MathJax(use_left_union[r]), MathJax(use_right_diff[r]))));

                        sw.WriteLine("$$");

                        //------------------------------------------------------------  式を簡約化する。
                        sw.WriteLine("<h5>式を簡約化する。</h5>");
                        sw.WriteLine("$$");
                        sw.Write("= ");

                        sw.WriteLine(string.Join(" + ", from r in prop_union
                                                            select string.Format(@"\delta^{{ {0} }} \cdot {1}",
                                                            MathJax(use_left_union[r]), MathJax(use_right_diff_simple[r]))));

                        sw.WriteLine("$$");

                        Dictionary<Reference, Variable> delta_fnc = prop_union.ToDictionary(r => r, r => new Variable("δ_" + r.Name, null, null));

                        Term result = SimplifyExpression(Add((from r in prop_union
                            select Mul(new Apply(new Reference(delta_fnc[r]), (from i in r.Indexes select i.Clone(null)).ToArray()), use_right_diff_simple[r])).ToArray()).Clone(null));
                        
                        sw.WriteLine("<pre><b>");
                        sw.WriteLine("double δ_" + left.Name + "(" + string.Join(", ", from i in left.Indexes select "int " + i.ToString()) + "){");
                        sw.WriteLine("\treturn " + result.ToString() + ";");
                        sw.WriteLine("}");
                        sw.WriteLine("</b></pre>");
                    }

                    //Debug.WriteLine(left.ToString() + " : " + string.Join(" ", from v in prop select v.Name) + " t-1:" + string.Join(" ", from v in prop_t_1 select v.Name));
                }
            }

            WriteMathJax(sw);
        }

        Reference MakeDifferentialReference(Variable t_var, Apply t_sub_1, Variable fld, Reference[] vref) {
            int dim_cnt = (fld.TypeVar as ArrayType).DimCnt;
            Term[] idxes = new Term[dim_cnt];

            for (int i = 0; i < dim_cnt; i++) {

                // 変数参照の添え字のリスト
                var vidx_ref = from r in vref where r.Indexes[i] is Reference select r.Indexes[i] as Reference;
                if (vidx_ref.Any()) {
                    // 変数参照の添え字がある場合

                    // スカラーの変数参照の添え字のリスト
                    var vi2 = from idx in vidx_ref where idx.Indexes == null select idx;
                    if (vi2.Any()) {
                        // スカラーの変数参照の添え字がある場合

                        // foreachのループ変数を参照する添え字のリスト
                        var vi3 = from idx in vi2 where idx.VarRef.ParentVar is ForEach select idx;
                        if (vi3.Any()) {
                            // foreachのループ変数を参照する添え字がある場合

                            idxes[i] = vi3.First();
                        }
                        else {
                            // foreachのループ変数を参照する添え字がない場合

                            // LINQのループ変数を参照する添え字のリスト
                            var vi4 = from idx in vi2 where idx.VarRef.ParentVar is LINQ select idx;
                            if (vi4.Any()) {
                                // LINQのループ変数を参照する添え字がある場合

                                idxes[i] = vi4.First();
                            }
                            else {
                                // LINQのループ変数を参照する添え字がない場合

                                throw new Exception();
                            }
                        }
                    }
                    else {

                        throw new Exception();
                    }
                }
                else {

                    // 関数適用の添え字のリスト
                    var vidx_app = from r in vref where r.Indexes[i] is Apply select r.Indexes[i] as Apply;
                    if (vidx_app.Any()) {
                        // 関数適用の添え字がある場合

                        // t - 1 以外の関数適用の添え字のリスト
                        var vidx_app2 = from ap in vidx_app where !ap.Eq(t_sub_1) select ap;
                        if (vidx_app2.Any()) {
                            // t - 1 以外の関数適用の添え字がある場合

                            if (vref.Count() == 1) {
                                // 変数参照が1個の場合

                                idxes[i] = vidx_app2.First();
                            }
                            else {
                                // 変数参照が複数の場合

                                throw new Exception();
                            }
                        }
                        else {
                            // すべて t - 1 の関数適用の添え字の場合

                            idxes[i] = vidx_app.First();
                        }
                    }
                    else {
                        // 関数適用の添え字がない場合

                        throw new Exception();
                    }
                }
            }

            for (int i = 0; i < dim_cnt; i++) {
                Debug.Assert(idxes[i] != null);

                foreach (Reference r1 in vref) {
                    if (!r1.Indexes[i].Eq(idxes[i])) {
                        // 添え字が等しくない場合

                        if (idxes[i] is Reference) {
                            // 基本の添え字が変数参照の場合

                            Reference idx_ref1 = idxes[i] as Reference;
                            if (idx_ref1.VarRef == t_var) {
                                // 基本の添え字が t の場合

                                if (r1.Indexes[i].Eq(t_sub_1)) {
                                    // 対象の添え字が t-1 の場合

                                }
                                else {
                                    // 対象の添え字が t-1 でない場合

                                    throw new Exception();
                                }
                            }
                            else {
                                // 基本の添え字が t でない場合

                                if (r1.Indexes[i] is Reference) {
                                    // 対象の添え字が変数参照の場合

                                    Reference idx_ref2 = r1.Indexes[i] as Reference;

                                    if (idx_ref1.VarRef.Domain != null && idx_ref1.VarRef.Domain.Eq(idx_ref2.VarRef.Domain)) {
                                        // 領域が同じ場合

                                    }
                                    else {
                                        // 領域が違う場合

                                        Debug.WriteLine("IDX domain ERR: {0} != {1}", idx_ref1.ToString(), idx_ref2.ToString());
                                    }
                                }
                                else {
                                    // 対象の添え字が変数参照でない場合

                                    throw new Exception();
                                }
                            }
                        }
                        else {
                            // 添え字が変数参照でない場合

                            throw new Exception();
                        }
                    }
                }
            }

            return new Reference(fld.Name, fld, idxes);
        }

        void WriteMathJax(StringWriter sw) {
/*

*/

            string head = @"<!DOCTYPE html>

<html lang=""en"" xmlns=""http://www.w3.org/1999/xhtml"">
<head>
    <meta charset=""utf-8"" />
    <title>MyML</title>
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
            File.WriteAllText(html_dir + "\\MathJax.html", head + sw.ToString() + "\r\n</body></html>", Encoding.UTF8);
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

        // u[i, j, k] = (from p in Range(H) from q in Range(H) select x[i + p, j + q] * h[p, q, k]).Sum() + b[k];
        // m = i + p   : 0 <= i <= I - 1   0 <= p <= H - 1
        //  i = m - p  : 0 <= m - p <= I - 1    m - I + 1 <= p <= m  max(0, m - I + 1) <= p <= min(H - 1, m)
        // n = j + q   : 0 <= j <= J - 1   0 <= q <= H - 1
        //  j = n - q  : 0 <= n - q <= J - 1    n - J + 1 <= q <= n  max(0, n - J + 1) <= q <= min(H - 1, n)
        // 

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

                        return Zero;
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


        void plus_linq(Reference ref1, List<int> plus_linq_dim, out LINQ lnq1, out LINQ lnq2) {
            List<Apply> i_plus_p = new List<Apply>();
            List<Variable> lnq_vars = new List<Variable>();

            Statement stmt = ParentStatement(ref1);
            Debug.Assert(stmt is Assignment);
            Assignment asn = stmt as Assignment;

            Term[] u_idxes = (from t in asn.Left.Indexes select t.Clone()).ToArray();
            Term[] x_idxes = (from t in ref1.Indexes select t.Clone()).ToArray();
            Dictionary<Reference, Term> subst_tbl = new Dictionary<Reference, Term>();

            foreach (int dim in plus_linq_dim) {
                Apply app = ref1.Indexes[dim] as Apply;

                var v = from a in i_plus_p where a.Eq(app) select a;
                if (! v.Any()) {

                    i_plus_p.Add(app);

                    Variable for_var1 = (app.Args[0] as Reference).VarRef;
                    Variable linq_var1 = (app.Args[1] as Reference).VarRef;

                    string name = for_var1.Name + "" + linq_var1.Name;
                    Apply for_var2_domain = new Apply(new Reference(DomainFnc), new Term[] { new Reference(ref1.VarRef), new Number(dim) });
                    Variable for_var2 = new Variable(name, for_var1.TypeVar, for_var2_domain);

                    x_idxes[dim] = new Reference(for_var2);

                    Apply start = Add(Sub(for_var2, MaxRange(for_var1.Domain)), One);
                    Reference end = new Reference(for_var2);
                    Apply linq_var2_domain = Intersect(linq_var1.Domain, Range(start, end));

                    Variable linq_var2 = new Variable(linq_var1.Name, linq_var1.TypeVar, linq_var2_domain);

                    lnq_vars.Add(linq_var2);

                    var vv = asn.Left.Indexes.Select((t, i) => new { Val = t, Idx = i }).Where(p => p.Val is Reference && (p.Val as Reference).VarRef == for_var1).Select(p => p.Idx);
                    if (vv.Any()) {

                        int i = vv.First();

                        Debug.Assert(u_idxes[i] is Reference);

                        Apply sub2 = Sub(for_var2, linq_var1);

                        subst_tbl.Add(u_idxes[i] as Reference, sub2);

                        u_idxes[i] = sub2;
                    }
                }
                else {

                    x_idxes[dim] = ref1.Indexes[dim].Clone();
                }
            }

            Reference u = new Reference(asn.Left.Name, asn.Left.VarRef, u_idxes);
            Apply diff1 = Diff(new Reference(EFnc), u);

            Reference x = new Reference(ref1.Name, ref1.VarRef, x_idxes);
            Apply diff2 = Diff(u.Clone(), x);

            Apply mul2 = Mul(diff1, diff2);

            lnq1 = new LINQ(lnq_vars.ToArray(), mul2, new Reference(SumFnc));


            Term u_right = Subst(asn.Right.Clone(), subst_tbl);

            Apply diff1_2 = diff1.Clone();
            Apply diff2_2 = Diff(u_right, x.Clone());
            Apply mul3 = Mul(diff1_2, diff2_2);

            lnq2 = new LINQ(lnq_vars.ToArray(), mul3, new Reference(SumFnc));
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
            return new Apply(new Reference(SubFnc), VariableToReference(args));
        }

        public int[] Range(int n) {
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

                return Zero;
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
                    return One;
                }
                else {
                    return Zero;
                }
            }
            else if (t1 is Number) {
                // 数値の場合

                return Zero;
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
                else if (app.Function.VarRef == SubFnc) {
                    // 減算の場合

                    return Sub(diffs);
                }
                else if (app.Function.VarRef == MulFnc) {
                    // 乗算の場合

                    Term[] args = new Term[app.Args.Length];
                    foreach(int i in Range(app.Args.Length)) {
                        args[i] = Mul((from j in Range(app.Args.Length) select (i == j ? diffs[i] : app.Args[j].Clone(var_tbl))).ToArray());
                    }

                    return Add(args);
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
                    return Zero;
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

                        List<Term> args2 = new List<Term>();
                        List<double> constants = new List<double>();

                        if (app.IsAdd() || app.IsSub()) {

                            double sign;
                            for(int i = 0; i < app.Args.Length; i++) {
                                Term t2 = app.Args[i];

                                if(app.IsSub() && i != 0) {
                                    // 減算で最初でない場合

                                    // 符号を負にする。
                                    sign = -1;
                                }
                                else {
                                    // 加算か最初の場合

                                    // 符号を正にする。
                                    sign = 1;
                                }

                                if(t2 is Number) {
                                    // 引数が定数の場合

                                    Number n = t2.ToNumber();

                                    // すでに定数があるか調べる。
                                    var vn = args2.Select((x, idx) => new { Val = x, Idx = idx }).Where(p => p.Val is Number).Select(p => p.Idx);

                                    if (vn.Any()) {
                                        // すでに定数がある場合

                                        int idx = vn.First();
                                        args2[idx] = new Number( args2[idx].ToNumber().Value + (sign * n.Value) );
                                    }
                                    else {

                                        args2.Add(new Number(sign * n.Value));
                                        constants.Add(1);
                                    }
                                }
                                else if (t2.IsAdd()) {
                                    // 引数が加算の場合

                                    foreach (Term t3 in t2.ToApply().Args) {

                                        args2.Add(t3);
                                        constants.Add(sign);
                                    }
                                }
                                else if (t2.IsSub()) {
                                    // 引数が減算の場合

                                    for (int j = 0; j < t2.ToApply().Args.Length; j++) {
                                        if (j != 0) {
                                            // 最初でない場合

                                            // 符号を反転する。
                                            sign *= -1;
                                        }

                                        args2.Add(t2.ToApply().Args[j] );
                                        constants.Add(sign);
                                    }
                                }
                                else {
                                    // 引数が加算や減算でない場合

                                    args2.Add(t2);
                                    constants.Add(sign);
                                }
                            }

                            for(int i = 0; i < args2.Count; i++) {

                                for (int j = i + 1; j < args2.Count;) {
                                    if(args2[i].Eq(args2[j])) {
                                        // 同じ項がある場合

                                        // 係数を加算する。
                                        constants[i] += constants[j];

                                        // 同じ項を取り除く
                                        args2.RemoveAt(j);
                                        constants.RemoveAt(j);

                                        if(constants[i] == 0) {
                                            // 係数が0の場合

                                            // 項を取り除く
                                            args2.RemoveAt(i);
                                            constants.RemoveAt(i);

                                            j--;
                                        }
                                    }
                                    else {
                                        // 同じ項がない場合

                                        j++;
                                    }
                                }
                            }

                            // すでに定数があるか調べる。
                            var vplus = constants.Select((x, idx) => new { Val = x, Idx = idx }).Where(p => 0<= p.Val).Select(p => p.Idx);

                            switch (args2.Count) {
                            case 0:
                                ret = Zero;
                                return true;

                            case 1:
                                args2[0].Parent = app.Parent;
                                ret = args2[0];

                                return true;
                            }

                        }

                        if (app.IsSub() && args1[0].IsAdd() && args1[1] is Number) {
                            // (t + 1) - 1

                            Debug.Assert(args1.Length == 2);

                            Apply app3 = args1[0] as Apply;
                            Number n = args1[1] as Number;

                            app.Function = app3.Function;
                            List<Term> v = new List<Term>(app3.Args);
                            v.Add(new Number(- n.Value));
                            args1 = v.ToArray();
                        }
                        /*
                        */

                        if (app.IsAdd() || app.IsMul()) {
                            foreach(Term t in args1) {
                                if(t is Apply && (t as Apply).Function.VarRef == app.Function.VarRef) {
                                    args2.AddRange((t as Apply).Args);
                                }
                                else if(! (t is Number)) {
                                    args2.Add(t);
                                }
                            }

                            Number[] ns = (from t in args1 where t is Number select t as Number).ToArray();
                            if (ns.Any()) {
                                if (app.IsAdd()) {

                                    double d = (from x in ns select x.Value).Sum();
                                    if (d != 0) {

                                        args2.Add(new Number(d));
                                    }
                                }
                                else {

                                    double d = (from x in ns select x.Value).Aggregate((x,y) => x * y);
                                    if(d == 0) {

                                        ret = Zero;

                                        return true;
                                    }
                                    else if (d != 1) {

                                        args2.Insert(0, new Number(d));
                                    }
                                }
                            }

                            switch (args2.Count) {
                            case 0:
                                if (app.IsAdd()) {

                                    ret = Zero;
                                }
                                else if (app.IsMul()) {

                                    ret = One;
                                }
                                else {

                                    Debug.Assert(false);
                                }
                                return true;

                            case 1:
                                args2[0].Parent = app.Parent;
                                ret = args2[0];

                                return true;
                            }
                        }
                        else {
                            args2 = new List<Term>(args1);
                        }

                        Dictionary<Variable, Variable> var_tbl = new Dictionary<Variable, Variable>();
                        Apply app2 = new Apply(app.Function.Clone(var_tbl), args2.ToArray());
                        app2.Parent = app.Parent;
                        ret = app2;

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

                    if (app.Args.Length == 1) {

                        s = app.Function.Name + " " + MathJax(app.Args[0]);
                    }
                    else {

                        if(app.Function.VarRef == MulFnc) {

                            s = string.Join(@" \cdot ", from x in app.Args select MathJax(x));
                        }
                        else {

                            s = string.Join(" " + app.Function.Name + " ", from x in app.Args select MathJax(x));
                        }
                    }
                    if(s.IndexOf("0 + 0 + [0 \\cdot s_{t}^{j} + wI_{j} \\cdot 1] + 0") != -1) {
                        //MathJax(t1);
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

        void Navi(object obj, NaviAction before, NaviAction after = null) {
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
            }
            else if (obj is Statement) {
                if (obj is Assignment) {
                    Assignment asn = obj as Assignment;
                    Navi(asn.Left, before, after);
                    Navi(asn.Right, before, after);
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
            }
            else if (obj is Statement) {
                if (obj is Assignment) {
                    Assignment asn = obj as Assignment;
                    asn.Left = NaviRep(asn.Left, before, after) as Reference;
                    asn.Right = NaviRep(asn.Right, before, after) as Term;

                    asn.Left.Parent = obj;
                    asn.Right.Parent = obj;
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