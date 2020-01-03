import React from "react"
import { graphql } from "gatsby"
import BlogLink from "../components/blog/blogLink"
import Layout from "../components/layout"

const BlogsPage = ({
  data: {
    allMarkdownRemark: { edges },
  },
}) => {
  const Posts = edges
    .filter(edge => !!edge.node.frontmatter.date) // You can filter your posts based on some criteria
    .map(edge => <BlogLink key={edge.node.id} post={edge.node} />)

  return <Layout>{Posts}</Layout>
}

export default BlogsPage

export const pageQuery = graphql`
  query {
    allMarkdownRemark(sort: { order: DESC, fields: [frontmatter___date] }) {
      edges {
        node {
          id
          excerpt(pruneLength: 250)
          frontmatter {
            date(formatString: "MMMM DD, YYYY")
            path
            title
          }
        }
      }
    }
  }
`