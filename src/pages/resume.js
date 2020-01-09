import React from "react";
import Layout from "../components/layout";
import SEO from "../components/seo";
import Resume from "../components/resume";

export default function() {
    return (
        <Layout>
            <SEO lang="en" title="Resume" />
            <div style={{ minHeight: "600px" }}>
                <Resume />
            </div>
        </Layout>
    );
}
