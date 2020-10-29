import React from "react";
import SectionTitle from "./sectiontitle";
import { StaticQuery, graphql } from "gatsby";
import SocialLinks from "./sociallinks";
import ResumePDF from "./resume-pdf";

class Resume extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            submitDisabled: false
        };

        this.showContactForm = false;

        if (this.props.contact.api_url === "") {
            this.showContactForm = false;
        }
    }

    render() {
        return (
            <section id="resume" className="container">
                <div className="section-title">
                    <SectionTitle title="RESUME" />
                </div>
                <ResumePDF />

            </section>

        );
    }
}

export default () => (
    <StaticQuery
        query={graphql`
            query {
                site {
                    siteMetadata {
                        contact {
                            api_url
                            description
                            mail
                            phone
                            address
                        }
                    }
                }
            }
        `}
        render={data => <Resume contact={data.site.siteMetadata.contact} />}
    />
);
