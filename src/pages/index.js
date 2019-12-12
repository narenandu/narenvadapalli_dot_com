import React from "react"
import { Link } from "gatsby"

import Layout from "../components/layout"
import Image from "../components/image"
import SEO from "../components/seo"

const IndexPage = () => (
  <Layout>
    <SEO title="NarenVadapalliHome" />
    <h1> Hi Peeps </h1> <p> Welcome to the world of NarenVadapalli </p>{" "}
    <div class="grid-container">
      <div class="grid-item"><Link to="/portfolio/"> Portfolio </Link>{" "}</div>
      <div class="grid-item"><Link to="/blog/"> Blog </Link>{" "}</div>
    </div>
    <div
      style={{
        maxWidth: `500px`,
        marginBottom: `1.45rem`,
      }}
    >
      <Image />
    </div>{" "}

  </Layout>
)

export default IndexPage
