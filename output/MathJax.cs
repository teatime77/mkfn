using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace MkFn {
    public partial class MkFn {

        /*
            MathJaxを含むHTMLファイルを書く。
        */
        void WriteMathJax(StringWriter sw, string file_name) {
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
            if (!Directory.Exists(html_dir)) {

                Directory.CreateDirectory(html_dir);
            }
            File.WriteAllText(html_dir + "\\" + file_name + ".html", head + sw.ToString() + "\r\n</body></html>", Encoding.UTF8);
        }


        /*
            係数を含んだMathJaxのテキストを返す。
        */
        string MathJax(Term t1) {
            OutputLanguage = Language.MathJax;

            if (!(t1 is Number)) {
                // 数値定数でない場合

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

        /*
            係数を含まない本体のMathJaxのテキストを返す。
        */
        string MathJaxBody(Term t1) {
            if (t1 is Reference) {
                // 変数参照の場合

                Reference r1 = t1 as Reference;
                if (r1.Indexes == null) {
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
                // 数値定数の場合

                Number n = t1 as Number;

                return n.Value.ToString();
            }
            else if (t1 is Apply) {
                // 関数適用の場合

                Apply app = t1 as Apply;

                if ("+-*/%".Contains(app.Function.Name[0])) {
                    // 演算子の場合

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
                    // 演算子でない場合

                    string name = app.Function.Name;

                    if (app.Function.VarRef == DiffFnc) {

                        if (MathJaxDelta && app.Args[0] is Reference && (app.Args[0] as Reference).VarRef == EFnc) {

                            return string.Format(@"\delta^{{ {0} }}", MathJax(app.Args[1]));
                        }
                        string arg0 = MathJax(app.Args[0]);
                        if (app.Args[0] is Apply && !Char.IsLetter((app.Args[0] as Apply).Function.Name[0])) {

                            arg0 = "(" + arg0 + ")";
                        }
                        return string.Format(@"\frac{{ \partial {0} }}{{ \partial {1} }}", arg0, MathJax(app.Args[1]));
                    }
                    if (app.Function.VarRef == σ_prime) {
                        name = "σ'";
                    }
                    if (app.Function.VarRef == tanh_prime) {
                        name = "tanh'";
                    }
                    return name + "(" + string.Join(", ", from x in app.Args select MathJax(x)) + ")";
                }
            }
            else if (t1 is LINQ) {
                // LINQの場合

                LINQ lnq = t1 as LINQ;

                string s = "";

                Debug.Assert(lnq.Variables.All(v => Term.IsRange(v.Domain)));

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
    }
}